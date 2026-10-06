import chromadb
from datetime import datetime
import uuid

# Create local database
client = chromadb.PersistentClient(
    path="./speaknote_db"
)

# Create collection
collection = client.get_or_create_collection(
    name="speaknote_notes"
)


def save_note(note, title="Untitled Note", topic="General"):

    note_id = str(uuid.uuid4())

    collection.add(
        documents=[note],
        ids=[note_id],
        metadatas=[{
            "title": title,
            "topic": topic,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }]
    )


def search_notes(query):

    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas"]
    )

    return results


def get_all_notes():

    results = collection.get(
        include=["documents", "metadatas"]
    )

    return results

def delete_note(note_id):

    collection.delete(
        ids=[note_id]
    )