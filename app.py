# MindMate - AI Mental Health Companion
import streamlit as st
import sqlite3
from datetime import datetime
import os
import io
import tempfile
from groq import Groq
import base64
import time
import json
from collections import Counter

# ✅ Must be FIRST Streamlit command
st.set_page_config(page_title="🧘 MindMate", layout="wide")

# ✅ Initialize Groq client
@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error("🚨 Please set GROQ_API_KEY in Streamlit secrets or environment variables")
        st.stop()
    return Groq(api_key=api_key)

client = get_groq_client()

# ✅ Text-to-Speech using gTTS
def text_to_speech(text):
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='en')
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error(f"⚠️ TTS error: {e}")
        return None


# ✅ Database setup
def init_db():
    conn = sqlite3.connect('mindmate.db')
    c = conn.cursor()
    # chats table
    c.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_message TEXT,
        assistant_message TEXT,
        mood TEXT
    )""")
    # journals table
    c.execute("""
    CREATE TABLE IF NOT EXISTS journals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        entry TEXT
    )""")
    # settings table for small key/value persistence
    c.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

def insert_chat(user_msg, bot_msg, mood):
    conn = sqlite3.connect('mindmate.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (timestamp, user_message, assistant_message, mood) VALUES (?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_msg, bot_msg, mood))
    conn.commit()
    conn.close()
def insert_journal(entry):
    conn = sqlite3.connect('mindmate.db')
    c = conn.cursor()
    c.execute("INSERT INTO journals (date, entry) VALUES (?, ?)",
              (datetime.now().strftime("%Y-%m-%d"), entry))
    conn.commit()
    conn.close()

def get_chat_history():
    conn = sqlite3.connect('mindmate.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, user_message, assistant_message, mood FROM chats ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return data

def get_journal_history():
    conn = sqlite3.connect('mindmate.db')
    c = conn.cursor()
    c.execute("SELECT date, entry FROM journals ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return data


# --- small settings persistence helpers ---------------------------------
def set_setting(key, value):
    """Persist a JSON-serializable setting value into the settings table."""
    conn = sqlite3.connect('mindmate.db')
    c = conn.cursor()
    try:
        store_val = json.dumps(value)
    except Exception:
        store_val = str(value)
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, store_val))
    conn.commit()
    conn.close()


def get_setting(key, default=None):
    """Return a Python object if stored as JSON, otherwise return the raw string or default."""
    conn = sqlite3.connect('mindmate.db')
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = c.fetchone()
    conn.close()
    if not row:
        return default
    val = row[0]
    try:
        return json.loads(val)
    except Exception:
        return val


def voice_allowed():
    """Return True when voice is enabled in session state and persisted settings."""
    persisted = get_setting('voice_enabled', False)
    return bool(st.session_state.get('voice', False)) and bool(persisted)


def safe_rerun():
    """Try to rerun the Streamlit script in a way that's compatible across versions.
    Falls back to a browser reload if st.experimental_rerun is unavailable.
    """
    try:
        # preferred if available
        getattr(st, 'experimental_rerun')()
    except Exception:
        # fallback: inject a small JS reload; this usually triggers a page refresh
        try:
            st.markdown("<script>window.location.reload()</script>", unsafe_allow_html=True)
        except Exception:
            # last resort: set a session flag so UI can react next run
            st.session_state['_needs_reload'] = True


# cheerful voice wrapper
def tts_cheerful(text):
    # small wrapper to adjust wording for cheerfulness; keep gTTS call
    cheerful_text = text.rstrip('.') + '. ' + 'You’re doing great! Keep going with a smile.'
    return text_to_speech(cheerful_text)

# ✅ Voice Transcription
def transcribe_audio(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        with open(path, "rb") as file:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=file
            )
        os.unlink(path)
        return result.text
    except Exception as e:
        st.error(f"🎙️ Transcription error: {e}")
        return None

# ✅ Streamlit UI
st.title("🧠 MindMate: Your AI Mental Health Companion")

if 'voice' not in st.session_state:
    st.session_state.voice = False

# Sidebar control: explicit Enable Voice gesture
with st.sidebar:
    st.markdown("## Preferences")
    persisted_voice = get_setting('voice_enabled', False)
    col_a, col_b = st.columns([2,1])
    with col_a:
        en = st.button("Enable Voice 🔊")
    with col_b:
        dis = st.button("Disable Voice 🔇")

    if en:
        st.session_state.voice = True
        try:
            set_setting('voice_enabled', True)
            st.success("Voice enabled — future audio will autoplay when possible.")
        except Exception:
            st.warning("Could not persist voice setting, but voice is enabled this session.")

    if dis:
        st.session_state.voice = False
        try:
            set_setting('voice_enabled', False)
            st.info("Voice disabled — audio will not autoplay.")
        except Exception:
            st.warning("Could not persist voice setting, but voice is disabled this session.")

    if persisted_voice and not st.session_state.voice:
        # reflect persisted preference in session state without requiring a click
        st.session_state.voice = True

# --- Onboarding / Name personalization ----------------------------------
profile = get_setting('user_profile', None)
with st.sidebar.expander("Your Profile", expanded=True):
    if profile:
        name = profile.get('name') if isinstance(profile, dict) else None
        pronouns = profile.get('pronouns') if isinstance(profile, dict) else None
        baseline = profile.get('baseline_mood') if isinstance(profile, dict) else None
        st.markdown(f"**Welcome back, {name or 'friend'}!**")
        st.write(f"Pronouns: {pronouns or '—'}\nBaseline mood: {baseline or '—'}")
        if st.button("Edit profile"):
            profile = None
            set_setting('user_profile', None)
            safe_rerun()
    else:
        st.markdown("### Tell me a bit about you")
        with st.form(key='profile_form'):
            name_in = st.text_input('What should I call you?', placeholder='Your name')
            pronouns_in = st.selectbox('Pronouns', ['They/Them', 'She/Her', 'He/Him', 'Prefer not to say'])
            baseline_in = st.selectbox('Baseline mood (optional)', ['Calm', 'Happy', 'Anxious', 'Sad', 'Other'])
            submitted = st.form_submit_button('Save Profile')
            if submitted and name_in.strip():
                user_profile = {'name': name_in.strip(), 'pronouns': pronouns_in, 'baseline_mood': baseline_in}
                try:
                    set_setting('user_profile', user_profile)
                except Exception:
                    st.warning('Could not persist profile, but it will be available this session.')
                st.session_state['user_profile'] = user_profile
                st.success(f"Thanks — I'll call you {name_in.strip()} from now on!")
                safe_rerun()

# reflect profile into session state for immediate use
if 'user_profile' not in st.session_state:
    st.session_state['user_profile'] = profile

tabs = st.tabs(["💬 Chat", "📓 Journal", "🧘 Mindfulness", "🧘‍♀️ Yoga", "📊 Progress", "🎵 Music"])

# 💬 CHAT TAB
with tabs[0]:
    st.subheader("💬 Talk to MindMate")
    st.session_state.voice = st.checkbox("🔊 Voice Responses", value=st.session_state.voice)
    # personalize prompt with saved name when available
    profile = st.session_state.get('user_profile') or get_setting('user_profile', None)
    display_name = None
    if profile and isinstance(profile, dict):
        display_name = profile.get('name')
    placeholder_text = f"Hi {display_name}, how are you feeling today?" if display_name else "How are you feeling today?"
    user_message = st.text_input(placeholder_text, placeholder="Type your thoughts here...")

    audio_input = st.audio_input("🎙️ Record your message (optional)")
    if audio_input:
        text = transcribe_audio(audio_input.read())
        if text:
            user_message = text
            st.success(f"🗣️ Transcribed: {text}")

    if st.button("Send") and user_message.strip():
        try:
            mood_res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Classify mood in one word"},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2
            )
            mood = mood_res.choices[0].message.content.strip()

            bot_res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are MindMate, an empathetic mental health companion."},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7
            )
            bot_message = bot_res.choices[0].message.content.strip()

            insert_chat(user_message, bot_message, mood)

            st.info(f"🧠 Mood: **{mood}**")
            st.markdown(f"**You:** {user_message}")
            st.markdown(f"**MindMate:** {bot_message}")

            if voice_allowed():
                audio = text_to_speech(bot_message)
                if audio:
                    try:
                        # Convert raw audio bytes to base64 so we can embed as data URI
                        b64 = base64.b64encode(audio).decode('utf-8')
                        data_uri = f"data:audio/mp3;base64,{b64}"
                        # Render an autoplaying HTML audio element. This is triggered by the
                        # user's Send button click (user gesture) which improves browser autoplay
                        # permission chances. Keep Streamlit's st.audio as a fallback UI.
                        st.markdown(f'<audio src="{data_uri}" autoplay controls></audio>', unsafe_allow_html=True)
                    except Exception:
                        # If embedding fails for any reason, fall back to st.audio
                        st.audio(audio, format="audio/mp3")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    with st.expander("🕒 Chat History"):
        for ts, u, b, m in get_chat_history():
            st.markdown(f"🗓️ `{ts}` | 😌 **{m}**")
            st.markdown(f"- **You:** {u}")
            st.markdown(f"- **MindMate:** {b}")
            st.markdown("---")

# 📓 JOURNAL TAB
with tabs[1]:
    st.subheader("📓 Daily Journal")
    profile = st.session_state.get('user_profile') or get_setting('user_profile', None)
    display_name = profile.get('name') if profile and isinstance(profile, dict) else None
    journal_label = f"Write or record your thoughts, {display_name}:" if display_name else "Write or record your thoughts:"
    entry = st.text_area(journal_label, height=150)
    audio_journal = st.audio_input("🎙️ Record journal entry")
    if audio_journal:
        text = transcribe_audio(audio_journal.read())
        if text:
            entry = text
            st.success(f"🗣️ Transcribed: {text}")

    if st.button("Save Entry") and entry.strip():
        insert_journal(entry)
        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Provide short, positive encouragement"},
                    {"role": "user", "content": entry}
                ],
                temperature=0.7
            )
            note = res.choices[0].message.content.strip()
            st.success("✅ Saved!")
            st.info(f"💬 {note}")
            if voice_allowed():
                # read the encouragement aloud in a cheerful way
                audio = tts_cheerful(note)
                if audio:
                    try:
                        b64 = base64.b64encode(audio).decode('utf-8')
                        data_uri = f"data:audio/mp3;base64,{b64}"
                        st.markdown(f'<audio src="{data_uri}" autoplay controls></audio>', unsafe_allow_html=True)
                    except Exception:
                        st.audio(audio, format="audio/mp3")
        except:
            st.info("💬 Thanks for sharing! You're doing great 🌟")

    st.markdown("### 📅 Past Entries")
    for d, e in get_journal_history():
        st.markdown(f"🗓️ **{d}**: _{e}_")
        st.markdown("---")

# 🧘 MINDFULNESS
with tabs[2]:
    st.subheader("🧘 Mindfulness Exercises")
    exercise = st.selectbox("Choose an exercise:", ["4-4-4 Breathing", "Box Breathing", "Visualization"])

    # Helpful guidance text
    guidance_map = {
        "4-4-4 Breathing": "Inhale for 4s, hold for 4s, exhale for 4s. Repeat this cycle to calm your nervous system.",
        "Box Breathing": "Inhale 4s, hold 4s, exhale 4s, hold 4s. This helps reset your breath and attention.",
        "Visualization": "Close your eyes and picture a peaceful place. Notice colors, sounds, and sensations."
    }
    st.info(guidance_map[exercise])

    # session length and controls
    minutes = st.slider("Session length (minutes)", min_value=1, max_value=10, value=3)
    start = st.button("▶️ Start Guided Session")

    # Ensure session counters exist
    st.session_state.setdefault("mindfulness_sessions", 0)
    st.session_state.setdefault("meditation_sessions", 0)

    def _play_bytes(audio_bytes):
        try:
            b64 = base64.b64encode(audio_bytes).decode('utf-8')
            data_uri = f"data:audio/mp3;base64,{b64}"
            st.markdown(f'<audio src="{data_uri}" autoplay controls></audio>', unsafe_allow_html=True)
        except Exception:
            st.audio(audio_bytes, format="audio/mp3")

    if start:
        total_seconds = minutes * 60
        placeholder = st.empty()
        prog = st.progress(0)
        elapsed = 0

        if exercise in ("4-4-4 Breathing", "Box Breathing"):
            cycle_seconds = 12 if exercise == "4-4-4 Breathing" else 16  # Box has extra hold
            cycles = max(1, total_seconds // cycle_seconds)
            st.info(f"Starting {exercise} for {minutes} minute(s). {cycles} cycles will run.")

            for c in range(cycles):
                # Inhale
                placeholder.markdown(f"### Cycle {c+1}/{cycles}: Inhale (4s)")
                if voice_allowed():
                    audio = text_to_speech("Inhale")
                    if audio:
                        _play_bytes(audio)
                for i in range(4):
                    time.sleep(1)
                    elapsed += 1
                    prog.progress(min(elapsed / total_seconds, 1.0))

                # Hold
                placeholder.markdown("### Hold (4s)")
                if voice_allowed():
                    audio = text_to_speech("Hold")
                    if audio:
                        _play_bytes(audio)
                for i in range(4):
                    time.sleep(1)
                    elapsed += 1
                    prog.progress(min(elapsed / total_seconds, 1.0))

                # Exhale
                placeholder.markdown("### Exhale (4s)")
                if voice_allowed():
                    audio = text_to_speech("Exhale")
                    if audio:
                        _play_bytes(audio)
                for i in range(4):
                    time.sleep(1)
                    elapsed += 1
                    prog.progress(min(elapsed / total_seconds, 1.0))

        elif exercise == "Visualization":
            placeholder.markdown("### Visualization: Close your eyes and begin")
            if voice_allowed():
                audio = text_to_speech("Close your eyes and picture a calm scene. Breathe slowly and notice details.")
                if audio:
                    _play_bytes(audio)
            # simple countdown
            for s in range(total_seconds):
                time.sleep(1)
                prog.progress(min((s+1) / total_seconds, 1.0))

        placeholder.success("✔️ Session complete — well done!")
        # increment counters
        st.session_state.mindfulness_sessions = st.session_state.get("mindfulness_sessions", 0) + 1
        st.session_state.meditation_sessions = st.session_state.get("meditation_sessions", 0) + 1

# 🧘‍♀️ YOGA
with tabs[3]:
    st.subheader("🧘‍♀️ Guided Yoga Asanas")
    poses = {
        "Child's Pose": [
            "Come to your knees and sit back on your heels.",
            "Fold forward, forehead to the mat, arms extended or relaxed.",
            "Breathe here and feel the back release."
        ],
        "Easy Pose": [
            "Sit cross-legged with a straight spine.",
            "Rest hands on knees and soften your shoulders.",
            "Breathe slowly and focus on grounding."
        ],
        "Cat-Cow": [
            "Start on hands and knees.",
            "Inhale: drop belly and lift gaze (Cow).",
            "Exhale: round the back and tuck chin (Cat). Repeat."
        ]
    }
    pose = st.selectbox("Select Pose", list(poses.keys()))
    st.markdown("### Steps")
    for idx, step in enumerate(poses[pose], start=1):
        st.markdown(f"{idx}. {step}")

    hold_seconds = st.slider("Hold each step (seconds)", min_value=5, max_value=60, value=15)
    start_yoga = st.button("▶️ Start Yoga Flow")

    # Ensure yoga counter exists
    st.session_state.setdefault("yoga_sessions", 0)

    if start_yoga:
        placeholder = st.empty()
        prog = st.progress(0)
        total_steps = len(poses[pose])
        elapsed = 0
        total_seconds = total_steps * hold_seconds

        for i, step in enumerate(poses[pose], start=1):
            placeholder.markdown(f"### Step {i}/{total_steps}: {step}")
            if voice_allowed():
                audio = text_to_speech(step)
                if audio:
                    try:
                        b64 = base64.b64encode(audio).decode('utf-8')
                        data_uri = f"data:audio/mp3;base64,{b64}"
                        st.markdown(f'<audio src="{data_uri}" autoplay controls></audio>', unsafe_allow_html=True)
                    except Exception:
                        st.audio(audio, format="audio/mp3")
                # save last AI mood so Music tab can default to it
                try:
                    set_setting('last_ai_mood', mood)
                except Exception:
                    pass

            for s in range(hold_seconds):
                time.sleep(1)
                elapsed += 1
                prog.progress(min(elapsed / total_seconds, 1.0))

        placeholder.success("✔️ Yoga flow complete — great job!")
        st.session_state.yoga_sessions = st.session_state.get("yoga_sessions", 0) + 1



# 📊 PROGRESS
with tabs[4]:
    st.subheader("📊 Your Wellness Progress Overview")

    # Fetch data
    chats = get_chat_history()  # [(timestamp, user_message, assistant_message, mood)]
    journals = get_journal_history() if 'get_journal_history' in globals() else []  # optional
    meditation_sessions = st.session_state.get("meditation_sessions", 0)
    mindfulness_sessions = st.session_state.get("mindfulness_sessions", 0)
    yoga_sessions = st.session_state.get("yoga_sessions", 0)

    # Build mood counts and basic lists
    mood_counts = Counter()
    anxious_examples = []
    messages = []
    for ts, user_msg, assistant_msg, mood in chats:
        if mood:
            mood_counts[mood.lower()] += 1
        messages.append((ts, user_msg, assistant_msg, mood))
        # detect anxious mentions either in mood or message text
        text_lower = (user_msg or "").lower()
        if (mood and 'anx' in mood.lower()) or any(k in text_lower for k in ["anxious", "anxiety", "panic", "worried"]):
            anxious_examples.append((ts, user_msg))

    # First: Mindfulness & Yoga Summary, Journaling, Motivation (move these up)
    st.markdown("### 🌿 Mindfulness & Yoga Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="🧘‍♀️ Meditation Sessions", value=meditation_sessions)
        if meditation_sessions > 0:
            st.progress(min(meditation_sessions / 10, 1.0))

    with col2:
        st.metric(label="🌸 Mindfulness Practices", value=mindfulness_sessions)
        if mindfulness_sessions > 0:
            st.progress(min(mindfulness_sessions / 10, 1.0))

    with col3:
        st.metric(label="🧍‍♀️ Yoga Sessions", value=yoga_sessions)
        if yoga_sessions > 0:
            st.progress(min(yoga_sessions / 10, 1.0))

    st.divider()

    # Journaling Activity & Problem Progress
    st.markdown("### 📓 Journaling Activity & Problem Progress")
    if journals:
        st.success(f"You’ve journaled **{len(journals)} days** so far 🪶.")
        st.caption("Keep expressing your feelings regularly. It’s helping you grow 💖.")

        # simple 'problems solved' heuristic
        positive = ["resolved","improved","better","less","fixed","solved","helped","reduced","managed","overcame"]
        recent = journals[-10:]
        positive_count = 0
        for _, entry in recent:
            low = (entry or "").lower()
            if any(p in low for p in positive):
                positive_count += 1

        solved_pct = (positive_count / len(recent)) * 100 if recent else 0
        st.metric(label="Problems showing improvement (est.)", value=f"{solved_pct:.0f}%")
        st.markdown("**Recent journal excerpts indicating progress**")
        for d, entry in recent[-3:]:
            if any(p in (entry or "").lower() for p in positive):
                st.write(f"- `{d}` — {entry}")
    else:
        st.warning("You haven’t started journaling yet. Try writing one entry today!")

    st.divider()

    # Motivation & Streaks
    st.markdown("### 🔔 Motivation & Streaks")
    st.info("💬 Daily reminders are keeping you on track.\n🌸 You're doing your best — keep going!")
    st.caption("(Coming soon: streak tracking and daily achievement badges 🏆)")

    st.divider()

    # Now: Emotional Insights
    st.markdown("### 🧠 Emotional Insights")
    if mood_counts:
        total_moods = sum(mood_counts.values())
        st.write(f"You’ve shared feelings **{total_moods}** times.")

        # Top moods
        top = mood_counts.most_common(5)
        st.markdown("**Top moods**")
        for mood, count in top:
            st.text(f"{mood.capitalize()}: {count}")

        # Multi-mood trend: compute counts per mood across time buckets
        bins = 6
        bucket_size = max(1, len(messages) // bins)
        # collect unique moods
        unique_moods = list({m for _, _, _, m in messages if m})
        # create buckets
        bucket_series = {m: [] for m in unique_moods}
        for i in range(0, len(messages), bucket_size):
            part = messages[i:i+bucket_size]
            counts = Counter()
            for _, umsg, _, umood in part:
                if umood:
                    counts[umood.lower()] += 1
            for m in unique_moods:
                bucket_series[m].append(counts.get(m.lower(), 0))

        # Prepare a dataframe-like dict for st.line_chart
        chart_data = {}
        for m in unique_moods:
            chart_data[m.capitalize()] = bucket_series[m]
        if chart_data:
            st.markdown("**Mood trend over time**")
            st.line_chart(chart_data)
        else:
            st.info("Not enough mood-labeled chats to build a mood trend. Keep chatting to generate mood labels.")

        # show some example anxious messages (recent 3)
        if anxious_examples:
            st.markdown("**Recent anxious messages**")
            for ts, msg in anxious_examples[-3:]:
                st.write(f"- `{ts}` — {msg}")
    else:
        st.info("No mood data yet. Start chatting to track your emotional patterns .")


# 🎵 MUSIC
with tabs[5]:
    st.subheader("🎵 Mood Music")
    # Try to default the dropdown to the last AI-predicted mood (map to our available keys)
    options = ["Calm", "Sad", "Anxious", "Happy"]
    default_mood = None
    try:
        last = get_chat_history()
        if last:
            # most recent chat is first row (timestamp, user_message, assistant_message, mood)
            lmood = (last[0][3] or "").lower()
            if 'anx' in lmood or 'panic' in lmood or 'worri' in lmood:
                default_mood = "Anxious"
            elif 'sad' in lmood or 'down' in lmood:
                default_mood = "Sad"
            elif 'happy' in lmood or 'joy' in lmood or 'good' in lmood:
                default_mood = "Happy"
            elif 'calm' in lmood or 'relax' in lmood:
                default_mood = "Calm"
    except Exception:
        default_mood = None

    try:
        idx = options.index(default_mood) if default_mood in options else 0
    except Exception:
        idx = 0
    mood = st.selectbox("How are you feeling?", options, index=idx)
    # Use reliable direct MP3 URLs (SoundHelix samples) to ensure Streamlit can play them
    links = {
        "Calm": "scott-buckley-moonlight(chosic.com).mp3",
        "Sad": "Sonder(chosic.com).mp3",
        "Anxious": "New-Beginnings-chosic.com_.mp3",
        "Happy": "fm-freemusic-inspiring-optimistic-upbeat-energetic-guitar-rhythm(chosic.com).mp3"
    }
    # Play selected mood music (validate key)
    # Store the mood key in session state (not raw URL) so Play/Stop is mood-based
    if 'music_to_play' not in st.session_state:
        st.session_state.music_to_play = None

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔊 Play Mood Music"):
            st.session_state.music_to_play = mood
            # persist last-played mood
            try:
                set_setting('last_played_mood', mood)
            except Exception:
                pass
    with col2:
        if st.button("⏹️ Stop Music"):
            st.session_state.music_to_play = None

    # Resolve the file/URL for the currently selected mood and the stored play mood
    selected_url = links.get(mood)
    play_key = st.session_state.music_to_play
    play_url = links.get(play_key) if play_key else None

    def _render_audio_from_path(path):
        try:
            # If path looks like an http URL, embed directly. Otherwise read local bytes.
            if path.startswith('http'):
                st.markdown(f'<audio src="{path}" autoplay controls></audio>', unsafe_allow_html=True)
                st.audio(path, format='audio/mp3')
            else:
                # local file in workspace
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        data = f.read()
                    b64 = base64.b64encode(data).decode('utf-8')
                    data_uri = f"data:audio/mp3;base64,{b64}"
                    st.markdown(f'<audio src="{data_uri}" autoplay controls></audio>', unsafe_allow_html=True)
                    st.audio(data, format='audio/mp3')
                else:
                    # Fall back to Streamlit audio by passing the path (Streamlit may serve it)
                    st.audio(path, format='audio/mp3')
        except Exception:
            try:
                st.audio(path, format='audio/mp3')
            except Exception:
                st.warning("Unable to play audio for this mood.")

    # If a mood is currently playing, render autoplaying player for that mood
    if play_url:
        _render_audio_from_path(play_url)
    else:
        # Show the selected mood's player but don't autoplay
        if selected_url:
            try:
                if os.path.exists(selected_url):
                    with open(selected_url, 'rb') as f:
                        st.audio(f.read(), format='audio/mp3')
                else:
                    st.audio(selected_url, format='audio/mp3')
            except Exception:
                st.warning("Unable to render audio for the selected mood.")