import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
import sqlite3
from datetime import datetime
import os
import csv


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--save_metadata", action="store_true", help="Save metadata to CSV file.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    if args.save_metadata:
        for i in range(len(documents)):
            metadata = documents[i].metadata
            # Extract the source path
            source_path = metadata['source']

            # Extract the filename and file type
            filename = os.path.basename(source_path)
            file_type = os.path.splitext(filename)[1][1:]  # get the file extension without the dot
            
            metadata_population_csv(file_name=filename, file_type=file_type, file_size=0)
    
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    # a PDF is an array of documents, where each document contains the page content and metadata with page number.
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
def metadata_population_csv(file_name, file_type, file_size):
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get the file information
    file_name = file_name
    file_type = file_type
    file_size = os.path.getsize(f"{DATA_PATH}/{file_name}")# / (1024 * 1024)  # Convert to MB

    # Define the CSV file path
    csv_file_path = os.path.join(os.path.dirname(__file__), "metadata.csv")

    # Check if the CSV file already exists
    csv_file_exists = os.path.exists(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, "a", newline="") as file:
        writer = csv.writer(file)
        
        # Get the last index in the CSV file
        # last_index = 0
        if csv_file_exists:
            with open(csv_file_path, "r") as file:
                reader = csv.reader(file)
                last_index = sum(1 for _ in reader) - 1

        # Calculate the new index
        new_index = last_index + 1
        
        metadata = [new_index, current_datetime, file_name, file_type, file_size]

        # Write the index and metadata row
        writer.writerow([new_index] + metadata)
        # Write the header row if the file doesn't exist
        if not csv_file_exists:
            writer.writerow(["Index","Datetime", "File Name", "File Type", "File Size MB"])

        # Write the metadata row
        writer.writerow(metadata)

    print("✅ Metadata saved to CSV file.")

if __name__ == "__main__":
    main()
