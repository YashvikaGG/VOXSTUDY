import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import ollama

from rag import save_note, search_notes, get_all_notes, delete_note


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VoxStudy",
    page_icon="🎙️",
    layout="centered"
)


# --------------------------------------------------
# VOXSTUDY STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f8fc;
    }

    h1 {
        text-align: center;
        font-size: 44px !important;
        font-weight: 800 !important;
        margin-bottom: 0px;
    }

    [data-testid="stCaptionContainer"] {
        text-align: center;
        font-size: 16px;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px 18px;
        font-size: 15px;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    textarea {
        border-radius: 12px !important;
    }

    input {
        border-radius: 12px !important;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
    }

    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    hr {
        margin-top: 25px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎙️ VoxStudy")
st.caption("Turn your voice into clear, organized study notes using local AI.")

st.divider()


# --------------------------------------------------
# LOAD WHISPER MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )


model = load_model()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "notes" not in st.session_state:
    st.session_state.notes = ""

if "note_title" not in st.session_state:
    st.session_state.note_title = ""

if "note_topic" not in st.session_state:
    st.session_state.note_topic = ""

if "show_saved_notes" not in st.session_state:
    st.session_state.show_saved_notes = False

if "note_saved" not in st.session_state:
    st.session_state.note_saved = False

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "quiz" not in st.session_state:
    st.session_state.quiz = []

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False


# --------------------------------------------------
# 1. UPLOAD AUDIO
# --------------------------------------------------

st.subheader("🎤 1. Upload Audio")

audio_file = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav", "m4a", "mp4"]
)


if audio_file is not None:

    st.audio(audio_file)

    if st.button(
        "🎙️ Transcribe Audio",
        use_container_width=True
    ):

        suffix = os.path.splitext(audio_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(audio_file.read())
            audio_path = temp_file.name

        with st.spinner("🎧 Converting speech to text..."):

            segments, info = model.transcribe(audio_path)

            transcript = ""

            for segment in segments:
                transcript += segment.text + " "

        os.remove(audio_path)

        st.session_state.transcript = transcript.strip()

        st.session_state.notes = ""
        st.session_state.note_title = ""
        st.session_state.note_topic = ""
        st.session_state.note_saved = False
        st.session_state.flashcards = []
        st.session_state.quiz = []
        st.session_state.quiz_submitted = False

        st.success("Transcription completed!")


# --------------------------------------------------
# 2. TRANSCRIPT
# --------------------------------------------------

if st.session_state.transcript:

    st.divider()

    st.subheader("📝 2. Transcript")

    st.text_area(
        "Your transcription",
        st.session_state.transcript,
        height=200
    )


    # --------------------------------------------------
    # 3. GENERATE AI NOTES
    # --------------------------------------------------

    st.subheader("✨ 3. Generate AI Notes")

    if st.button(
        "✨ Generate Notes",
        use_container_width=True
    ):

        prompt = f"""
You are an intelligent note-taking assistant.

Analyze the following transcript and create clear, useful study notes.

Use exactly these sections:

## Short Summary
Write 2-3 simple sentences.

## Key Points
Give the most important points as bullet points.

## Organized Notes
Organize the information using suitable headings and bullet points.

Keep the answer concise and easy to understand.

Transcript:
{st.session_state.transcript}
"""

        with st.spinner("🤖 Creating notes with local AI..."):

            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.session_state.notes = response["message"]["content"]


        # --------------------------------------------------
        # GENERATE TITLE
        # --------------------------------------------------

        title_prompt = f"""
Create a short and clear title for the following lecture transcript.

Rules:
- Maximum 6 words
- No quotation marks
- Return only the title

Transcript:
{st.session_state.transcript}
"""

        title_response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": title_prompt
                }
            ]
        )

        st.session_state.note_title = (
            title_response["message"]["content"].strip()
        )


        # --------------------------------------------------
        # GENERATE TOPIC
        # --------------------------------------------------

        topic_prompt = f"""
Identify the main topic of this lecture.

Return only one short topic name.
Maximum 3 words.

Transcript:
{st.session_state.transcript}
"""

        topic_response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": topic_prompt
                }
            ]
        )

        st.session_state.note_topic = (
            topic_response["message"]["content"].strip()
        )

        st.session_state.note_saved = False

        st.success("AI notes generated successfully!")


# --------------------------------------------------
# 4. AI GENERATED NOTES
# --------------------------------------------------

if st.session_state.notes:

    st.divider()

    st.subheader("✨ 4. AI Generated Notes")

    with st.container(border=True):

        st.markdown(
            f"### 📚 {st.session_state.note_title}"
        )

        st.caption(
            f"🗂️ Topic: {st.session_state.note_topic}"
        )

        st.divider()

        st.markdown(
            st.session_state.notes
        )


# --------------------------------------------------
# 5. AI FLASHCARDS
# --------------------------------------------------

if st.session_state.notes:

    st.divider()

    st.subheader("🃏 5. AI Flashcards")

    if st.button(
        "🃏 Generate Flashcards",
        use_container_width=True
    ):

        flashcard_prompt = f"""
Create exactly 5 useful study flashcards from the notes below.

Return ONLY in this format:

Q1: question
A1: answer

Q2: question
A2: answer

Q3: question
A3: answer

Q4: question
A4: answer

Q5: question
A5: answer

Do not add any other text.

Notes:
{st.session_state.notes}
"""

        with st.spinner("🃏 Creating flashcards..."):

            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "user",
                        "content": flashcard_prompt
                    }
                ]
            )

            flashcard_text = response["message"]["content"]


        # --------------------------------------------------
        # PARSE FLASHCARDS
        # --------------------------------------------------

        flashcards = []

        lines = flashcard_text.splitlines()

        question = None

        for line in lines:

            line = line.strip()

            if line.startswith("Q") and ":" in line:

                question = line.split(":", 1)[1].strip()

            elif line.startswith("A") and ":" in line and question:

                answer = line.split(":", 1)[1].strip()

                flashcards.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

                question = None

        st.session_state.flashcards = flashcards

        st.success("Flashcards generated!")


    # --------------------------------------------------
    # DISPLAY FLASHCARDS
    # --------------------------------------------------

    if st.session_state.flashcards:

        for i, card in enumerate(
            st.session_state.flashcards
        ):

            with st.container(border=True):

                st.markdown(
                    f"### 🃏 Flashcard {i + 1}"
                )

                st.markdown(
                    f"**Question**"
                )

                st.write(
                    card["question"]
                )

                st.divider()

                st.markdown(
                    "**💡 Answer**"
                )

                st.write(
                    card["answer"]
                )


# --------------------------------------------------
# 6. AI QUIZ
# --------------------------------------------------

if st.session_state.notes:

    st.divider()

    st.subheader("🧠 6. AI Quiz")

    if st.button(
        "🧠 Generate Quiz",
        use_container_width=True
    ):

        quiz_prompt = f"""
Create exactly 5 multiple-choice questions from the notes below.

Use this exact format:

Q1: question
A: option
B: option
C: option
D: option
ANSWER: A

Q2: question
A: option
B: option
C: option
D: option
ANSWER: B

Q3: question
A: option
B: option
C: option
D: option
ANSWER: C

Q4: question
A: option
B: option
C: option
D: option
ANSWER: D

Q5: question
A: option
B: option
C: option
D: option
ANSWER: A

Rules:
- Questions must be based only on the notes.
- Each question must have exactly 4 options.
- Only one option should be correct.
- Do not add explanations.
- Do not add any extra text.

Notes:
{st.session_state.notes}
"""

        with st.spinner("🧠 Creating quiz..."):

            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "user",
                        "content": quiz_prompt
                    }
                ]
            )

            quiz_text = response["message"]["content"]


        # --------------------------------------------------
        # PARSE QUIZ
        # --------------------------------------------------

        quiz = []

        current = {}

        for line in quiz_text.splitlines():

            line = line.strip()

            if line.startswith("Q") and ":" in line:

                if (
                    current.get("question")
                    and current.get("answer")
                ):

                    quiz.append(current)

                current = {
                    "question": line.split(
                        ":", 1
                    )[1].strip()
                }

            elif line.startswith("A:"):

                current["A"] = line[2:].strip()

            elif line.startswith("B:"):

                current["B"] = line[2:].strip()

            elif line.startswith("C:"):

                current["C"] = line[2:].strip()

            elif line.startswith("D:"):

                current["D"] = line[2:].strip()

            elif line.startswith("ANSWER:"):

                current["answer"] = (
                    line.split(":", 1)[1]
                    .strip()
                )


        if (
            current.get("question")
            and current.get("answer")
        ):

            quiz.append(current)


        st.session_state.quiz = quiz

        st.session_state.quiz_submitted = False

        st.success("Quiz generated!")


    # --------------------------------------------------
    # DISPLAY QUIZ
    # --------------------------------------------------

    if st.session_state.quiz:

        st.write(
            "Choose the best answer for each question."
        )

        user_answers = []

        for i, q in enumerate(
            st.session_state.quiz
        ):

            with st.container(border=True):

                st.markdown(
                    f"### Question {i + 1}"
                )

                st.write(
                    q["question"]
                )

                answer = st.radio(
                    "Choose an answer:",
                    [
                        "Select an option",
                        f"A. {q['A']}",
                        f"B. {q['B']}",
                        f"C. {q['C']}",
                        f"D. {q['D']}"
                    ],
                    key=f"quiz_question_{i}"
                )

                if answer == "Select an option":
                    user_answers.append("")
                else:
                    user_answers.append(
                        answer[0]
                    )


        # --------------------------------------------------
        # SUBMIT QUIZ
        # --------------------------------------------------

        if st.button(
            "✅ Submit Quiz",
            use_container_width=True
        ):

            score = 0

            for i, q in enumerate(
                st.session_state.quiz
            ):

                selected = user_answers[i]

                correct = (
                    q["answer"]
                    .strip()
                    .upper()
                )

                if selected == correct:
                    score += 1


            st.session_state.quiz_submitted = True

            total = len(
                st.session_state.quiz
            )

            percentage = (
                score / total
            ) * 100


            st.success(
                f"🎉 Your Score: {score}/{total}"
            )

            st.write(
                f"📊 Percentage: {percentage:.0f}%"
            )


# --------------------------------------------------
# SAVE NOTE
# --------------------------------------------------

if st.session_state.notes:

    st.divider()

    st.subheader("💾 7. Save This Note")

    with st.container(border=True):

        st.markdown(
            f"### 📚 {st.session_state.note_title}"
        )

        st.caption(
            f"🗂️ Topic: {st.session_state.note_topic}"
        )

        if not st.session_state.note_saved:

            if st.button(
                "💾 Save Note",
                use_container_width=True
            ):

                save_note(
                    st.session_state.notes,
                    st.session_state.note_title,
                    st.session_state.note_topic
                )

                st.session_state.note_saved = True

                st.success(
                    "Note saved successfully! 🧠"
                )

        else:

            st.success(
                "✅ This note is already saved."
            )


# --------------------------------------------------
# START NEW
# --------------------------------------------------

if (
    st.session_state.transcript
    or st.session_state.notes
):

    st.divider()

    if st.button(
        "🔄 Start New",
        use_container_width=True
    ):

        st.session_state.transcript = ""
        st.session_state.notes = ""
        st.session_state.note_title = ""
        st.session_state.note_topic = ""
        st.session_state.note_saved = False
        st.session_state.flashcards = []
        st.session_state.quiz = []
        st.session_state.quiz_submitted = False

        st.rerun()


# --------------------------------------------------
# ASK MY NOTES
# --------------------------------------------------

st.divider()

st.subheader("🔎 Ask My Notes")

question = st.text_input(
    "Ask a question about your saved notes"
)


if st.button(
    "🔎 Search Notes",
    use_container_width=True
) and question:

    with st.spinner(
        "🔍 Searching your notes..."
    ):

        results = search_notes(question)


        if (
            results["documents"]
            and results["documents"][0]
        ):

            documents = results["documents"][0]

            metadatas = results["metadatas"][0]

            context = ""


            for i, document in enumerate(
                documents
            ):

                metadata = metadatas[i]


                if metadata:

                    title = metadata.get(
                        "title",
                        "Untitled Note"
                    )

                    date = metadata.get(
                        "date",
                        ""
                    )

                else:

                    title = "Older Note"

                    date = ""


                context += f"""
Source: {title}
Date: {date}

{document}

--------------------
"""


            prompt = f"""
Answer the question using only the information
provided in the notes below.

Notes:
{context}

Question:
{question}

Give a clear and concise answer.
"""


            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )


            answer = response[
                "message"
            ]["content"]


            st.subheader("💡 Answer")

            with st.container(border=True):

                st.write(answer)


        else:

            st.info(
                "No relevant notes found."
            )


# --------------------------------------------------
# SAVED NOTES
# --------------------------------------------------

st.divider()

st.subheader("📚 My Saved Notes")


if st.button(
    "📖 View Saved Notes",
    use_container_width=True
):

    st.session_state.show_saved_notes = True


if st.session_state.show_saved_notes:

    saved_notes = get_all_notes()

    documents = saved_notes["documents"]

    metadatas = saved_notes["metadatas"]

    ids = saved_notes["ids"]


    if documents:

        for i, note in enumerate(
            documents
        ):

            metadata = metadatas[i]


            if metadata:

                title = metadata.get(
                    "title",
                    "Untitled Note"
                )

                date = metadata.get(
                    "date",
                    ""
                )

                topic = metadata.get(
                    "topic",
                    "General"
                )

            else:

                title = "Older Note"

                date = ""

                topic = "General"


            with st.expander(
                f"📚 {title}"
            ):

                if date:

                    st.caption(
                        f"📅 {date}"
                    )

                st.caption(
                    f"🗂️ Topic: {topic}"
                )

                st.write(note)


                if st.button(
                    "🗑️ Delete This Note",
                    key=f"delete_{ids[i]}"
                ):

                    delete_note(
                        ids[i]
                    )

                    st.session_state.show_saved_notes = True

                    st.success(
                        "Note deleted successfully! 🗑️"
                    )

                    st.rerun()


    else:

        st.info(
            "No saved notes yet."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "🎙️ VoxStudy • Local AI Study Assistant"
)