import pathlib

from dotenv import load_dotenv
from langchain.text_splitter import TokenTextSplitter
from llama_index import download_loader

from lib import database


def main():
    pdf_file_path = pathlib.Path("./hirokisakabe_post.pdf")

    CJKPDFReader = download_loader("CJKPDFReader")
    pdf_loader = CJKPDFReader()

    documents = pdf_loader.load_data(file=pdf_file_path)

    # llama_indexのフォーマットからLangChainのフォーマットに変換
    documents = [doc.to_langchain_format() for doc in documents]

    token_text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = token_text_splitter.split_documents(documents)

    db = database.get_database()

    db.add_documents(split_documents)


if __name__ == "__main__":
    load_dotenv()

    main()
