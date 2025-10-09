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
import random
from statistics import mean

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


# -------------------- Mood analysis helpers --------------------
NEGATIVE_KEYWORDS = ['sad', 'anx', 'depress', 'tired', 'low', 'down', 'worri', 'panic']

def last_n_days_moods(n=14):
    """Return the list of moods (strings) from the last n chats (most recent first)."""
    chats = get_chat_history()  # already returns newest-first
    moods = []
    seen_dates = set()
    for ts, _, _, mood in chats:
        # only one mood per date
        date = ts.split(' ')[0]
        if date in seen_dates:
            continue
        seen_dates.add(date)
        moods.append((date, (mood or '').lower()))
        if len(moods) >= n:
            break
    return moods

def count_negative_days(moods):
    """Count how many of the provided (date,mood) tuples are negative."""
    c = 0
    for _, mood in moods:
        if any(k in (mood or '') for k in NEGATIVE_KEYWORDS):
            c += 1
    return c

def negative_streak(moods):
    """Compute the current consecutive negative-day streak (from most recent going backwards)."""
    streak = 0
    for _, mood in moods:
        if any(k in (mood or '') for k in NEGATIVE_KEYWORDS):
            streak += 1
        else:
            break
    return streak

def positivity_score(mood):
    """Return a small numeric positivity score for a mood string (0..1).
    Simple heuristic: negative keywords -> 0.2, neutral/unknown -> 0.5, positive keywords -> 0.9
    """
    if not mood:
        return 0.5
    m = mood.lower()
    if any(k in m for k in NEGATIVE_KEYWORDS):
        return 0.2
    if any(k in m for k in ['happy', 'joy', 'good', 'calm', 'relax']):
        return 0.9
    return 0.5

def gentle_depression_check_and_prompt():
    """If 10+ out of last 14 days are negative, show a gentle AI prompt and TTS suggestion."""
    moods = last_n_days_moods(14)
    if not moods:
        return False
    neg = count_negative_days(moods)
    if neg >= 10:
        # get user's name if available
        profile = st.session_state.get('user_profile') or get_setting('user_profile')
        name = (profile.get('name') if isinstance(profile, dict) else None) or 'friend'
        msg = f"Hey {name} 🌷, I noticed you’ve been feeling quite low recently. Would you like to try a calming meditation or talk about what’s been heavy lately?"
        st.warning(msg)
        if voice_allowed():
            audio = tts_cheerful(msg)
            if audio:
                try:
                    b64 = base64.b64encode(audio).decode('utf-8')
                    data_uri = f"data:audio/mp3;base64,{b64}"
                    st.markdown(f'<audio src="{data_uri}" autoplay controls></audio>', unsafe_allow_html=True)
                except Exception:
                    st.audio(audio, format='audio/mp3')
        return True
    return False

# Small pool of positive affirmations
AFFIRMATIONS = [
    "You are doing your best — and that is enough.",
    "Small steps add up. Be kind to yourself today.",
    "Breathe. You are here, and you matter.",
    "Every day is a new chance to feel a little better."
]

def show_affirmation_and_checkin():
    """Show a random affirmation and a gentle check-in if last mood was negative."""
    aff = random.choice(AFFIRMATIONS)
    st.info(f"💛 {aff}")
    # gentle check-in if last mood negative
    moods = last_n_days_moods(1)
    if moods and any(k in (moods[0][1] or '') for k in NEGATIVE_KEYWORDS):
        st.info("Gentle check-in: I noticed your last mood looked a bit low. Would you like a short breathing exercise?")


def show_daily_affirmation_if_needed():
    """Show a daily affirmation once per calendar day. Returns True when shown."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        last = get_setting('last_affirmation_date', None)
        if last == today:
            return False
        # show affirmation and gentle check-in
        aff = random.choice(AFFIRMATIONS)
        st.info(f"💛 {aff}")
        moods = last_n_days_moods(1)
        if moods and any(k in (moods[0][1] or '') for k in NEGATIVE_KEYWORDS):
            st.info("Gentle check-in: I noticed your last mood looked a bit low. Would you like a short breathing exercise?")
        try:
            set_setting('last_affirmation_date', today)
        except Exception:
            pass
        return True
    except Exception:
        # degrade to session-only behavior
        aff = random.choice(AFFIRMATIONS)
        st.info(f"💛 {aff}")
        return True

# ✅ Streamlit UI
st.title("🧠 MindMate: Your AI Mental Health Companion")

if 'voice' not in st.session_state:
    st.session_state.voice = False

# Show a daily affirmation once per calendar day (persisted). Fall back to session-only.
if 'daily_affirmation_shown' not in st.session_state:
    # show_daily_affirmation_if_needed will persist the date; it returns True when it displayed
    shown = False
    try:
        shown = show_daily_affirmation_if_needed()
    except Exception:
        # fallback to session-only
        show_affirmation_and_checkin()
        shown = True
    st.session_state['daily_affirmation_shown'] = bool(shown)

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
    st.subheader("📊 Your Wellness Progress — Friendly Overview")

    # Run daily affirmation and check-ins at top
    show_affirmation_and_checkin()
    # run gentle depression check (may speak/txt)
    gentle_depression_check_and_prompt()

    # Data pulls
    moods = last_n_days_moods(14)  # list of (date, mood)
    journals = get_journal_history() if 'get_journal_history' in globals() else []
    meditation_sessions = st.session_state.get("meditation_sessions", 0)
    mindfulness_sessions = st.session_state.get("mindfulness_sessions", 0)
    yoga_sessions = st.session_state.get("yoga_sessions", 0)

    # Weekly Mood Overview (average positivity over the last 14 days)
    positivity_scores = [positivity_score(m) for _, m in moods] if moods else []
    weekly_avg = (mean(positivity_scores) if positivity_scores else 0.5)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### 💭 Weekly Mood Overview")
        st.progress(weekly_avg)
        st.caption(f"Average positivity (last {len(positivity_scores)} days): {weekly_avg:.2f}")

    # Negative Mood Streak
    with col2:
        st.markdown("### 🌧️ Negative Mood Streak")
        streak = negative_streak(moods)
        st.metric(label="Days in a row feeling low", value=streak)

    st.divider()

    # Mindfulness Stats
    st.markdown("### 🧘 Mindfulness & Yoga")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Meditation sessions", meditation_sessions)
    with c2:
        st.metric("Mindfulness practices", mindfulness_sessions)
    with c3:
        st.metric("Yoga sessions", yoga_sessions)

    st.divider()

    # Journaling activity
    st.markdown("### 📓 Journaling")
    st.write(f"Journaled days: {len(journals)}")
    if journals:
        st.write("Recent entries:")
        for d, e in journals[:3]:
            st.write(f"- `{d}` — {e[:150]}{'...' if len(e) > 150 else ''}")

    st.divider()

    # Encouragement generated by the model (short)
    st.markdown("### 💌 Encouragement")
    try:
        prompt = (
            "You are a gentle wellness coach. Produce a 1-line encouraging message acknowledging small progress for a user trying to feel better."
        )
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"system","content":"You are kind and concise."}, {"role":"user","content":prompt}],
            temperature=0.6
        )
        encourage = res.choices[0].message.content.strip()
    except Exception:
        encourage = "You’re doing great. Keep taking small steps each day — it adds up."

    st.info(encourage)

    # --- Mood trend chart (restore detailed trend view) ---
    # Build a simple trend chart using the last messages we collected above
    chats = get_chat_history()
    messages = []
    mood_counts = Counter()
    for ts, user_msg, assistant_msg, mood in chats:
        if mood:
            mood_counts[mood.lower()] += 1
        messages.append((ts, user_msg, assistant_msg, mood))

    if messages:
        bins = 6
        bucket_size = max(1, len(messages) // bins)
        unique_moods = list({m for _, _, _, m in messages if m})
        bucket_series = {m: [] for m in unique_moods}
        for i in range(0, len(messages), bucket_size):
            part = messages[i:i+bucket_size]
            counts = Counter()
            for _, umsg, _, umood in part:
                if umood:
                    counts[umood.lower()] += 1
            for m in unique_moods:
                bucket_series[m].append(counts.get(m.lower(), 0))

        chart_data = {}
        for m in unique_moods:
            chart_data[m.capitalize()] = bucket_series[m]
        if chart_data:
            st.markdown("### 📈 Mood trend over time")
            try:
                # line_chart will import pandas under the hood; guard against environments
                # where pandas isn't available during quick import checks.
                st.line_chart(chart_data)
            except Exception as e:
                # Don't let a missing optional dependency crash the whole app during imports.
                st.warning("Could not render trend chart (optional dependency missing). The rest of the app is available.")
                # log to console for debugging
                print('Chart render skipped:', e)
    else:
        st.info("Not enough mood-labeled chats to build a mood trend. Keep chatting to generate mood labels.")


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