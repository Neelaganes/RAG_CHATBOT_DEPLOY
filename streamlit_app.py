import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import chromadb


load_dotenv()
groq_api_key = os.getenv("groq_api_key")


data_folder=os.path.join(os.path.dirname(__file__), "_data") 

st.set_page_config(
    page_title="Novatech Internal Assistant", 
    page_icon="message",
    layout="centered")

st.title("Novatech Internal Assistant")
st.caption("Ask anything about company products, policies, and procedures.")
st.divider()

def load_rag():

    if not os.path.exists(data_folder):
        st.error("Data folder not found at " + data_folder)
        st.stop()

    client=OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    chroma=chromadb.EphemeralClient()

    try:
        chroma.delete_collection("docs")
    except Exception:
        pass

    collection=chroma.create_collection("docs")

    all_chunks=[]
    all_ids=[]
    all_metadatas=[]
    chunk_id=0

    for file_name in os.listdir(data_folder):
        if not file_name.endswith(".txt"):
            continue

        with open(os.path.join(data_folder, file_name), "r") as f:
            text=f.read()

        for para in text.strip().split("\n\n"):
            para=para.strip()

            if len(para) < 50:
                continue

            if para.startswith("===="):
                continue

            all_chunks.append(para)
            all_ids.append(f"chunk_{chunk_id}")
            all_metadatas.append({"source": file_name,"text": para})
            chunk_id += 1

    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas        
    )

    return collection, client, chunk_id

with st.spinner("Loading and indexing comapny documents..."):
    collection,groq_client,total_chunks=load_rag()

st.success(f"Ready - {total_chunks} document chunks indexed.")
st.divider()



if "messages" not in st.session_state:
    st.session_state.messages=[]



def ask_rag(question:str):

    results=collection.query(query_texts=[question],n_results=3)

    chunks=results["documents"][0]
    sources=[m["source"] for m in results["metadatas"][0]]

    context="\n\n---\n\n".join(chunks)

    messages=[
        {"role":"system",
            "content":(
             "you are a helpful Novatech company assistant."
             " answer questions using only provided context."
              "if the context doesn't contain the enough information,"
              "say i don't have enough information to answer this "
              "Be concise and direct."
            )
        },{
          "role":"user",
          "content":f"context: \n{context} \n\n Question:{question}"
      }
    ]

    response=groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2,
    )

    answer=response.choices[0].message.content

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "chunks": list(zip(chunks,sources))
    }

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


        if msg["role"]=="assistant" and msg.get("sources"):

            st.caption(f"Sources: {', '.join(msg['sources'])}")

            with st.expander("See context chunks used for answer"):
                for i,(chunk,source) in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i} - {source}):**")
                    st.markdown(chunk)
            

question=st.chat_input("Ask a question about Novatech products or policies")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role":"user","content":question})

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result=ask_rag(question)

            st.markdown(result["answer"])
            st.caption(f"Sources: {', '.join(result['sources'])}")

            with st.expander("See context chunks used for answer"):
                for i,(chunk,source) in enumerate(result["chunks"], 1):
                    st.markdown(f"**Chunk {i} - {source}):**")
                    st.markdown(chunk)
    st.session_state.messages.append({
        "role":"assistant",
        "content":result["answer"],
        "sources":result["sources"],
        "chunks":result["chunks"]
        })