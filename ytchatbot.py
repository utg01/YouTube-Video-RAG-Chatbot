from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone,ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder
import streamlit as st

load_dotenv()
google_api = st.text_input(
    "Enter Google API Key",
    type="password"
)
if google_api:
    os.environ["GOOGLE_API_KEY"] = google_api
api=os.environ["GOOGLE_API_KEY"]

import requests
from langchain_classic.schema import Document
if api:
    embeddings=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=api
    )
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api
    )
    pc=Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )


reranker = CrossEncoder(
    ".models/reranker"
)

st.header("Welcome to the Youtube Chatbot")
st.subheader("Enter your youtube video link, gemini api key and chat about anything from the youtube video ")
st.subheader("Disclaimer:")
st.text("The chatbot currently works for videos with english transcripts available")
st.warning("Please enter your Google API key to continue")
@tool
def Create_embeddings(link: str) -> dict:
    """
    Use this tool when the user provides a YouTube video link.

    This tool:
    - Fetches the English transcript of the YouTube video
    - Splits it into chunks and generates vector embeddings
    - Stores the embeddings in a Pinecone vector database

    Args:
        link: The full YouTube video URL

    Returns on success:
        {
            "status": "success",
            "video_title": <title>,
            "index_name": <pinecone index name>
        }

    Returns on failure:
        Error message string
    """

    try:
        import re
        if "watch?v=" in link:
            video_id = link.split("watch?v=")[1].split("&")[0]
        else:
            video_id = link
        oembed_url = (
            f"https://www.youtube.com/oembed"
            f"?url=https://www.youtube.com/watch?v={video_id}&format=json"
        )

        title_response = requests.get(
            oembed_url,
            timeout=20
        )

        if title_response.status_code != 200:
            return "Could not fetch video title"

        video_title = title_response.json().get(
            "title",
            "youtube-video"
        )
        # fetching transcript
        params = {
            "engine": "youtube_video_transcript",
            "v": video_id,
            "language_code": "en",
            "api_key": os.getenv("SERPAPI_API_KEY")
        }

        response = requests.get(
            "https://serpapi.com/search.json",
            params=params,
            timeout=40
        )

        if response.status_code != 200:
            return f"Transcript fetch failed: {response.status_code}"

        transcript_data = response.json().get(
            "transcript",
            []
        )

        if not transcript_data:
            return "English transcript not available"

        #formating transcript

        transcript = " ".join(
            chunk.get("snippet", "")
            for chunk in transcript_data
            if chunk.get("snippet")
        ).strip()

        if not transcript:
            return "English transcript not available"

        #creating documents

        docs = [
            Document(
                page_content=transcript,
                metadata={
                    "title": video_title,
                    "source": link
                }
            )
        ]

        # splitting

        all_splits = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_documents(docs)

        # index name
        index_name = f"yt-{video_id}"
        index_name = re.sub(
            r'[^a-z0-9-]',
            '-',
            index_name.lower()
        )
        index_name = re.sub(
            r'-+',
            '-',
            index_name
        ).strip('-')

        index_name = index_name[:45]

        # getting embedding dimensions

        embedding_dimension = len(
            embeddings.embed_query("hello")
        )
        #creating index
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            existing_dimension = pc.describe_index(
                index_name
            ).dimension
            # deleting wrong-dimension index automatically
            if existing_dimension != embedding_dimension:
                pc.delete_index(index_name)

        # recreate after deletion check
        existing_indexes = pc.list_indexes().names()

        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        #storing embeddings

        PineconeVectorStore.from_documents(
            documents=all_splits,
            embedding=embeddings,
            index_name=index_name
        )
        return {
            "status": "success",
            "video_title": video_title,
            "index_name": index_name
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Create_embeddings Error: {str(e)}"
    

def chunk_reranker(question,docs):
    pairs=[
        (question,doc.page_content)for doc in docs
    ]
    scores=reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True)
    top_docs=scored_docs[:3]
    #global cache_chunks
    #cache_chunks=top_docs[:2]
    return top_docs
    


def context_retriever_from_docs(top_docs):
    context = "\n\n".join(
        doc.page_content
        for doc, score in top_docs)
    return context

def fresh_retrieval(question,index_name,k, fetch_k):
    vectorstore=PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":k,
            "fetch_k":fetch_k
        }
    )
    docs=retriever.invoke(question)
    top_chunks= chunk_reranker(question=question,docs=docs)
    return top_chunks

def cache_retrieval(question,cache_chunks):
    plain_docs = [doc for doc, score in cache_chunks]
    cache_top_chunks=chunk_reranker(question=question,docs=plain_docs)
    cache_relevant = [
        doc
        for doc, score in cache_top_chunks
        if score > threshold_score
    ]
    return cache_relevant
cache_chunks=[]
threshold_score=0.85

@tool
def Query_Handler(question: str, index_name: str) -> str:
    """
    Use this tool to answer any question about a YouTube video whose transcript has already been embedded.

    This tool:
    - Searches the Pinecone vector store for relevant transcript chunks
    - Uses a cross-encoder reranker to find the most relevant chunks
    - Returns the most relevant transcript context as a string

    Args:
        question: The user's question about the video
       index_name: The Pinecone index name returned by Create_embeddings

    Returns:
        A string containing the most relevant transcript context to answer the question.
        Use this context to form your final answer to the user.

    IMPORTANT: 
    - Only use this tool if Create_embeddings has already been called for this video in the current conversation.
    - Use the index_name returned by Create_embeddings — do not guess or fabricate it.
    - If the returned context does not contain the answer, say "I couldn't find this in the video." Do not hallucinate.
    """
    cache_relevant=cache_retrieval(question=question,cache_chunks=cache_chunks)
    if cache_relevant:
        fresh_chunks_with_score=fresh_retrieval(
            question=question,
            index_name=index_name,
            k=2,
            fetch_k=5)
        fresh_docs=[
            doc 
            for doc,score in fresh_chunks_with_score
        ]
        combined_docs=(fresh_docs + cache_relevant)
        new_scored = chunk_reranker(question=question, docs=combined_docs)
        cache_chunks.extend(new_scored)
        cache_chunks.sort(key=lambda x: x[1], reverse=True)
        del cache_chunks[10:]
        return context_retriever_from_docs(new_scored)
    else: 
        fresh_chunks_with_score=fresh_retrieval(question=question,index_name=index_name,k=5,fetch_k=10)
        fresh_docs=[
            doc 
            for doc,score in fresh_chunks_with_score
        ]
        new_scored = chunk_reranker(question=question, docs=fresh_docs)
        cache_chunks.extend(new_scored)
        cache_chunks.sort(key=lambda x: x[1], reverse=True)
        del cache_chunks[10:]
        return context_retriever_from_docs(new_scored)
        
system_prompt = SystemMessage("""
You are a YouTube video assistant. Your job is to answer questions about a YouTube video based on its transcript.

You have access to two tools:
1. Create_embeddings(link) — Call this when the user provides a YouTube link. This loads the transcript and stores it. It returns the index_name of the video — remember it for future queries.
2. Query_Handler(question, index_name) — Call this to answer questions about the video using the stored transcript.

Rules:
- Always call Create_embeddings first when a new video link is provided.
- Use the index_name returned by Create_embeddings in all subsequent Query_Handler calls.
- Only answer based on the transcript context returned by Query_Handler. Do not make up information.
- If the transcript doesn't contain the answer, say "I couldn't find this in the video."
- Keep answers concise and relevant.
""")
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [system_prompt]  

if api:
    tools=[Create_embeddings, Query_Handler]
    llm_with_tools=llm.bind_tools(tools)
    tools_map = {
        "Create_embeddings": Create_embeddings,
        "Query_Handler": Query_Handler
    }
    for message in st.session_state.conversation_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            if message.content:   
                with st.chat_message("assistant"):
                    st.markdown(message.content)
    humanmsg = st.chat_input("You:")
    if (humanmsg and humanmsg.lower() != "exit"):
        st.session_state.conversation_history.append(HumanMessage(content=humanmsg))
        response=llm_with_tools.invoke(st.session_state.conversation_history)
        st.session_state.conversation_history.append(AIMessage(content=response.content))
        while response.tool_calls:
            for tool_call in response.tool_calls:
                tool_tocall=tools_map[tool_call["name"]]
                result=tool_tocall.invoke(tool_call["args"])
                st.session_state.conversation_history.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )
            response = llm_with_tools.invoke(st.session_state.conversation_history)
            st.session_state.conversation_history.append(AIMessage(content=response.content))    
        humanmsg = st.chat_input("You: ")
    del st.session_state.conversation_history