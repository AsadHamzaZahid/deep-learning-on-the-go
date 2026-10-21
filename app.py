import pickle
import random
from pathlib import Path

import numpy as np
import streamlit as st
from keras.models import load_model
from keras.utils import pad_sequences

# ------------------------------------------------------------------ settings
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Artifacts" / "BiGru_model.keras"
TOKENIZER_PATH = BASE_DIR / "Artifacts" / "tokenizer.pkl"

MAX_LEN = 50  # same as training: pad_sequences(maxlen=50, padding="post", truncating="post")
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # dair-ai/emotion label order

st.set_page_config(page_title="How do you feel today?", page_icon="🌈", layout="centered")

# ------------------------------------------------------------------ content
# Short, simple, kid-friendly quotes (all original).
QUOTES = {
    "sadness": [
        "It's okay to cry. Tears help your heart feel lighter.",
        "Big feelings are visitors. They come, and then they go.",
        "Even rainy days help flowers grow.",
        "You are loved, even on your sad days.",
        "Take a deep breath. Tomorrow is a brand new page to color!",
    ],
    "joy": [
        "Your smile is a little piece of sunshine. Share it!",
        "Happy feels even better when you share it.",
        "Today is a great day to dance!",
        "You make the world brighter just by being you.",
        "Let's celebrate the little things!",
    ],
    "love": [
        "Love grows bigger every time you share it.",
        "A kind word is like a warm hug.",
        "Hearts are made for sharing.",
        "The people who love you are always in your corner.",
        "Being kind is the best superpower.",
    ],
    "anger": [
        "Breathe in slowly... now let it all out. Whoooosh!",
        "Count to ten. Your calm is stronger than your grumpy.",
        "Mad is a feeling, not a monster. You can tame it!",
        "Squeeze your fists, then let go. Feel the grumpy float away.",
        "It's okay to feel mad. Take a break and try again.",
    ],
    "fear": [
        "Being brave doesn't mean you're not scared. It means you try anyway.",
        "Scary feelings shrink when you talk about them.",
        "You are stronger than you think!",
        "Take one tiny step. Tiny steps are brave too!",
        "You are not alone. Someone is always ready to help.",
    ],
    "surprise": [
        "Wow! Life is full of fun surprises!",
        "Surprises turn a day into an adventure!",
        "Who knows what wonderful thing comes next?",
        "Open your eyes wide. There's so much to discover!",
        "Every day has a little surprise hiding in it.",
    ],
}

# sky color, ground color, shirt, hair, hair style, skin, floating decorations
THEME = {
    "sadness": dict(sky1="#CFE3F7", sky2="#EAF3FC", hill="#8FCB9B", shirt="#6FA8DC", hair="#6B4A2B",
                    style="bob", skin="#F5C9A0", floaters=["☁️", "🌈", "💧"], label="Feeling sad",
                    emoji="💙", color="#4A90D9"),
    "joy": dict(sky1="#FFF2B3", sky2="#FFE27A", hill="#7ED68B", shirt="#FF9F1C", hair="#3B2A1A",
                style="spiky", skin="#E8B98A", floaters=["⭐", "✨", "🎈"], label="Feeling happy",
                emoji="💛", color="#F4B400"),
    "love": dict(sky1="#FFD9E8", sky2="#FFEAF2", hill="#9ADBA0", shirt="#FF6F9C", hair="#D9822B",
                 style="pigtails", skin="#F7D2B0", floaters=["💗", "💖", "🌸"], label="Feeling loved",
                 emoji="💗", color="#EC5B8E"),
    "anger": dict(sky1="#FFD9CC", sky2="#FFEDE6", hill="#86C98F", shirt="#E4572E", hair="#2B2B2B",
                  style="cap", skin="#D9A273", floaters=["🍃", "☁️", "🌬️"], label="Feeling grumpy",
                  emoji="🧡", color="#E4572E"),
    "fear": dict(sky1="#D8D3F5", sky2="#ECE9FB", hill="#8CC7A0", shirt="#8E7CC3", hair="#8B5E3C",
                 style="curly", skin="#F1C79F", floaters=["🌙", "⭐", "🔦"], label="Feeling scared",
                 emoji="💜", color="#7E6BC4"),
    "surprise": dict(sky1="#CDEFE6", sky2="#E6F8F2", hill="#7ED6A0", shirt="#2BB5A0", hair="#C0392B",
                     style="bun", skin="#EBC09A", floaters=["🎈", "🎉", "✨"], label="Feeling surprised",
                     emoji="💚", color="#1FA58F"),
}

EXAMPLES = [
    "I won a prize today!",
    "I miss my best friend",
    "I heard a scary noise",
]


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Waking up the feelings robot...")
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict(text: str, model, tokenizer) -> np.ndarray:
    # IMPORTANT: no extra cleaning. The notebook trained on the raw text,
    # the Keras Tokenizer itself lowercases and strips punctuation.
    seq = tokenizer.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    try:
        probs = model.predict(x, verbose=0)[0]
    except Exception:
        # model was saved with a fixed batch size of 32 -> pad the batch and take row 0
        batch = np.repeat(x, 32, axis=0)
        probs = model.predict(batch, verbose=0)[0]
    return np.asarray(probs, dtype=float)


# ------------------------------------------------------------------ kid drawing
def hair_back(style: str, c: str) -> str:
    if style == "pigtails":
        return (f'<circle cx="168" cy="372" r="24" fill="{c}"/><circle cx="312" cy="372" r="24" fill="{c}"/>'
                f'<circle cx="168" cy="372" r="24" fill="none"/>')
    if style == "bob":
        return f'<rect x="172" y="290" width="136" height="120" rx="48" fill="{c}"/>'
    if style == "curly":
        return "".join(f'<circle cx="{x}" cy="{y}" r="22" fill="{c}"/>'
                       for x, y in [(180, 330), (200, 296), (240, 284), (280, 296), (300, 330), (176, 366), (304, 366)])
    return ""


def hair_front(style: str, c: str) -> str:
    fringe = (f'<path d="M176 346 Q172 288 240 288 Q308 288 304 346 Q274 318 240 322 Q206 318 176 346Z" fill="{c}"/>')
    if style == "spiky":
        return (f'<path d="M176 340 L180 282 L206 308 L224 270 L244 306 L266 272 L280 308 L304 284 L304 340 '
                f'Q274 318 240 322 Q206 318 176 340Z" fill="{c}"/>')
    if style == "bun":
        return f'<circle cx="240" cy="276" r="26" fill="{c}"/>' + fringe
    if style == "cap":
        return ('<path d="M174 336 Q176 276 240 276 Q304 276 306 336Z" fill="#3D5AFE"/>'
                '<rect x="168" y="328" width="144" height="14" rx="7" fill="#2A3FC4"/>'
                '<rect x="262" y="332" width="64" height="12" rx="6" fill="#2A3FC4"/>'
                '<circle cx="240" cy="276" r="8" fill="#FFD84D"/>')
    return fringe


def face(emotion: str) -> str:
    ink = "#2B2B2B"
    cheeks = '<circle cx="200" cy="388" r="10" fill="#FF9AA2" opacity=".55"/><circle cx="280" cy="388" r="10" fill="#FF9AA2" opacity=".55"/>'
    eyes_normal = (f'<g class="blink"><circle cx="214" cy="362" r="8" fill="{ink}"/><circle cx="266" cy="362" r="8" fill="{ink}"/>'
                   '<circle cx="217" cy="359" r="3" fill="#fff"/><circle cx="269" cy="359" r="3" fill="#fff"/></g>')
    if emotion == "sadness":
        return (eyes_normal + cheeks +
                f'<path d="M198 342 Q212 334 228 344" stroke="{ink}" stroke-width="4" fill="none" stroke-linecap="round"/>'
                f'<path d="M252 344 Q268 334 282 342" stroke="{ink}" stroke-width="4" fill="none" stroke-linecap="round"/>'
                f'<path d="M224 396 Q240 384 256 396" stroke="{ink}" stroke-width="5" fill="none" stroke-linecap="round"/>'
                '<path class="tear" d="M206 376 Q200 388 206 394 Q212 388 206 376Z" fill="#7EC8F7"/>')
    if emotion == "joy":
        return (eyes_normal + cheeks +
                f'<path d="M198 340 Q212 332 228 340" stroke="{ink}" stroke-width="4" fill="none" stroke-linecap="round"/>'
                f'<path d="M252 340 Q268 332 282 340" stroke="{ink}" stroke-width="4" fill="none" stroke-linecap="round"/>'
                '<path d="M214 382 Q240 418 266 382Z" fill="#8B1E2D"/>'
                '<path d="M226 396 Q240 406 254 396 Q240 400 226 396Z" fill="#FF7A8A"/>')
    if emotion == "love":
        return (f'<path d="M204 366 Q214 352 224 366" stroke="{ink}" stroke-width="5" fill="none" stroke-linecap="round"/>'
                f'<path d="M256 366 Q266 352 276 366" stroke="{ink}" stroke-width="5" fill="none" stroke-linecap="round"/>'
                '<circle cx="198" cy="388" r="12" fill="#FF6F9C" opacity=".6"/><circle cx="282" cy="388" r="12" fill="#FF6F9C" opacity=".6"/>'
                f'<path d="M222 390 Q240 410 258 390" stroke="{ink}" stroke-width="5" fill="none" stroke-linecap="round"/>')
    if emotion == "anger":
        return (eyes_normal +
                '<circle cx="198" cy="388" r="11" fill="#FF6B6B" opacity=".55"/><circle cx="282" cy="388" r="11" fill="#FF6B6B" opacity=".55"/>'
                f'<path d="M198 340 L228 350" stroke="{ink}" stroke-width="6" stroke-linecap="round"/>'
                f'<path d="M282 340 L252 350" stroke="{ink}" stroke-width="6" stroke-linecap="round"/>'
                f'<path d="M224 398 Q240 388 256 398" stroke="{ink}" stroke-width="5" fill="none" stroke-linecap="round"/>')
    if emotion == "fear":
        return ('<g class="blink"><circle cx="214" cy="360" r="13" fill="#fff" stroke="#2B2B2B" stroke-width="3"/>'
                '<circle cx="266" cy="360" r="13" fill="#fff" stroke="#2B2B2B" stroke-width="3"/>'
                f'<circle cx="214" cy="362" r="5" fill="{ink}"/><circle cx="266" cy="362" r="5" fill="{ink}"/></g>'
                f'<path d="M198 334 Q212 322 228 336" stroke="{ink}" stroke-width="4" fill="none" stroke-linecap="round"/>'
                f'<path d="M252 336 Q268 322 282 334" stroke="{ink}" stroke-width="4" fill="none" stroke-linecap="round"/>'
                f'<path d="M224 398 Q232 390 240 398 Q248 406 256 398" stroke="{ink}" stroke-width="4" fill="none" stroke-linecap="round"/>'
                '<path class="tear" d="M292 336 Q286 348 292 354 Q298 348 292 336Z" fill="#7EC8F7"/>')
    # surprise
    return ('<g class="blink"><circle cx="214" cy="360" r="11" fill="#fff" stroke="#2B2B2B" stroke-width="3"/>'
            '<circle cx="266" cy="360" r="11" fill="#fff" stroke="#2B2B2B" stroke-width="3"/>'
            f'<circle cx="214" cy="360" r="5" fill="{ink}"/><circle cx="266" cy="360" r="5" fill="{ink}"/></g>' + cheeks +
            f'<path d="M198 330 Q212 318 228 330" stroke="{ink}" stroke-width="4" fill="none" stroke-linecap="round"/>'
            f'<path d="M252 330 Q268 318 282 330" stroke="{ink}" stroke-width="4" fill="none" stroke-linecap="round"/>'
            f'<ellipse cx="240" cy="396" rx="10" ry="14" fill="#8B1E2D"/>')


def scene_html(emotion: str, quote: str) -> str:
    t = THEME[emotion]
    size = 30 if len(quote) <= 48 else 25 if len(quote) <= 72 else 22
    floaters = "".join(
        f'<text class="float f{i}" x="{x}" y="{y}" font-size="{s}">{e}</text>'
        for i, (e, x, y, s) in enumerate(zip(t["floaters"], [104, 350, 112], [346, 340, 292], [34, 34, 28]))
    )
    skin = t["skin"]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html,body {{ margin:0; background:transparent; font-family:'Fredoka','Comic Sans MS','Chalkboard SE',system-ui,sans-serif; }}
  .wrap {{ max-width:460px; margin:0 auto; }}
  svg {{ width:100%; height:auto; display:block; overflow:visible; }}
  .sway {{ transform-origin:240px 540px; animation: sway 3.6s ease-in-out infinite; }}
  .banner {{ transform-origin:240px 40px; animation: flap 2.4s ease-in-out infinite; }}
  .blink {{ transform-origin:240px 362px; animation: blink 4.5s infinite; }}
  .tear {{ animation: drip 2.2s ease-in infinite; }}
  .float {{ animation: bob 3.2s ease-in-out infinite; }}
  .f1 {{ animation-delay:-1s; }} .f2 {{ animation-delay:-2s; }}
  .armL {{ transform-origin:200px 452px; animation: waveL 2.4s ease-in-out infinite; }}
  .armR {{ transform-origin:280px 452px; animation: waveR 2.4s ease-in-out infinite; }}
  @keyframes sway {{ 0%,100% {{ transform:rotate(-1.6deg); }} 50% {{ transform:rotate(1.6deg); }} }}
  @keyframes flap {{ 0%,100% {{ transform:skewX(-1.2deg); }} 50% {{ transform:skewX(1.2deg); }} }}
  @keyframes blink {{ 0%,92%,100% {{ transform:scaleY(1); }} 96% {{ transform:scaleY(.1); }} }}
  @keyframes drip {{ 0% {{ transform:translateY(-4px); opacity:0; }} 30% {{ opacity:1; }} 100% {{ transform:translateY(16px); opacity:0; }} }}
  @keyframes bob {{ 0%,100% {{ transform:translateY(0); }} 50% {{ transform:translateY(-12px); }} }}
  @keyframes waveL {{ 0%,100% {{ transform:rotate(0); }} 50% {{ transform:rotate(-2deg); }} }}
  @keyframes waveR {{ 0%,100% {{ transform:rotate(0); }} 50% {{ transform:rotate(2deg); }} }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; }} }}
  .quote {{ width:100%; height:100%; display:flex; align-items:center; justify-content:center; text-align:center;
            font-weight:600; font-size:{size}px; line-height:1.18; color:#3a2a1a; padding:0 6px; box-sizing:border-box; }}
</style></head><body><div class="wrap">
<svg viewBox="0 0 480 560" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A child holding a banner that says: {quote}">
  <defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{t['sky1']}"/><stop offset="1" stop-color="{t['sky2']}"/></linearGradient></defs>
  <rect x="0" y="0" width="480" height="560" rx="36" fill="url(#sky)"/>
  {floaters}
  <g class="sway">
    <!-- poles -->
    <rect x="44" y="18" width="10" height="290" rx="5" fill="#B5793A"/>
    <rect x="426" y="18" width="10" height="290" rx="5" fill="#B5793A"/>
    <circle cx="49" cy="16" r="9" fill="#FFD84D"/><circle cx="431" cy="16" r="9" fill="#FFD84D"/>
    <!-- banner -->
    <g class="banner">
      <path d="M54 40 H426 V222 L402 238 L378 222 L354 238 L330 222 L306 238 L282 222 L258 238 L234 222 L210 238 L186 222 L162 238 L138 222 L114 238 L90 222 L66 238 L54 222Z" fill="#FFFDF5" stroke="{t['color']}" stroke-width="6" stroke-linejoin="round"/>
      <rect x="68" y="54" width="344" height="156" rx="14" fill="none" stroke="{t['color']}" stroke-width="3" stroke-dasharray="3 9" stroke-linecap="round"/>
      <foreignObject x="74" y="58" width="332" height="148"><div xmlns="http://www.w3.org/1999/xhtml" class="quote">{quote}</div></foreignObject>
    </g>
    <!-- kid -->
    <g class="armL"><path d="M206 460 Q120 420 54 292" stroke="{skin}" stroke-width="22" fill="none" stroke-linecap="round"/><circle cx="49" cy="290" r="16" fill="{skin}"/></g>
    <g class="armR"><path d="M274 460 Q360 420 426 292" stroke="{skin}" stroke-width="22" fill="none" stroke-linecap="round"/><circle cx="431" cy="290" r="16" fill="{skin}"/></g>
    <rect x="170" y="430" width="140" height="130" rx="44" fill="{t['shirt']}"/>
    <path d="M214 430 Q240 456 266 430Z" fill="#fff" opacity=".85"/>
    <rect x="224" y="412" width="32" height="26" rx="10" fill="{skin}"/>
    {hair_back(t['style'], t['hair'])}
    <circle cx="176" cy="366" r="12" fill="{skin}"/><circle cx="304" cy="366" r="12" fill="{skin}"/>
    <circle cx="240" cy="358" r="66" fill="{skin}"/>
    {hair_front(t['style'], t['hair'])}
    {face(emotion)}
  </g>
  <!-- hill -->
  <path d="M0 500 Q120 462 240 490 Q360 518 480 480 V524 Q480 560 444 560 H36 Q0 560 0 524Z" fill="{t['hill']}"/>
  <circle cx="70" cy="528" r="7" fill="#fff"/><circle cx="70" cy="528" r="3" fill="#FFD84D"/>
  <circle cx="410" cy="534" r="7" fill="#fff"/><circle cx="410" cy="534" r="3" fill="#FFD84D"/>
</svg></div></body></html>"""


def show_scene(html: str, height: int = 520):
    """Render the animated scene. Uses st.iframe (new) and falls back to components.html on older Streamlit.
    Only our own quote text goes into this HTML, never user input."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


def bars_html(probs: np.ndarray, top: int) -> str:
    rows = []
    for i in np.argsort(-probs):
        e = EMOTIONS[i]
        t = THEME[e]
        pct = probs[i] * 100
        rows.append(
            f'<div class="row"><span class="nm">{t["emoji"]} {e.capitalize()}</span>'
            f'<div class="track"><div class="fill" style="width:{max(pct, 1.5):.1f}%;background:{t["color"]}"></div></div>'
            f'<span class="pc">{pct:.0f}%</span></div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


# ------------------------------------------------------------------ page style
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stApp { font-family: 'Fredoka', 'Comic Sans MS', 'Chalkboard SE', sans-serif !important; }
.stApp { background-color:#FFF8E1;
  background-image: radial-gradient(#FFE08A 2.5px, transparent 2.6px), radial-gradient(#BDE5F8 2.5px, transparent 2.6px);
  background-size: 44px 44px; background-position: 0 0, 22px 22px; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 760px; padding-top: 2.2rem; }
.title { text-align:center; font-weight:700; font-size:2.7rem; line-height:1.1; color:#3a2a1a; margin:0 0 .2rem; }
.title span:nth-child(5n+1){color:#EC5B8E} .title span:nth-child(5n+2){color:#F4B400}
.title span:nth-child(5n+3){color:#1FA58F} .title span:nth-child(5n+4){color:#4A90D9} .title span:nth-child(5n){color:#7E6BC4}
.sub { text-align:center; font-size:1.15rem; color:#6b5440; margin-bottom:1.2rem; }
.stTextArea textarea { border:4px solid #3a2a1a !important; border-radius:22px !important; background:#fff !important;
  font-size:1.15rem !important; color:#3a2a1a !important; padding:14px 16px !important; }
.stTextArea textarea:focus { border-color:#4A90D9 !important; box-shadow:0 0 0 4px #BDE5F8 !important; }
.stTextArea label p { font-size:1.1rem; font-weight:600; color:#3a2a1a; }
.stButton > button { border:4px solid #3a2a1a; border-radius:999px; background:#FFD84D; color:#3a2a1a; font-weight:600;
  font-size:1.15rem; padding:.45rem 1.4rem; box-shadow:0 5px 0 #3a2a1a; transition:transform .08s, box-shadow .08s; }
.stButton > button:hover { background:#FFE27A; border-color:#3a2a1a; color:#3a2a1a; }
.stButton > button:active { transform:translateY(4px); box-shadow:0 1px 0 #3a2a1a; }
.stButton > button:focus-visible { outline:4px solid #4A90D9; outline-offset:3px; }
[class*="st-key-ex_"] button { background:#fff; border-width:3px; box-shadow:0 3px 0 #3a2a1a; font-size:.95rem; padding:.3rem .8rem; }
[class*="st-key-ex_"] button:hover { background:#FFF3C4; }
.verdict { text-align:center; font-weight:700; font-size:1.7rem; color:#3a2a1a; margin:.4rem 0 .1rem; }
.hint { text-align:center; color:#6b5440; font-size:1.05rem; margin-bottom:.8rem; }
.bars { background:#fff; border:4px solid #3a2a1a; border-radius:22px; padding:14px 16px; }
.bars .row { display:flex; align-items:center; gap:10px; margin:8px 0; }
.bars .nm { width:128px; font-weight:600; color:#3a2a1a; }
.bars .track { flex:1; height:18px; background:#F1EADB; border-radius:99px; overflow:hidden; }
.bars .fill { height:100%; border-radius:99px; }
.bars .pc { width:44px; text-align:right; font-weight:600; color:#3a2a1a; }
.err { background:#fff; border:4px solid #E4572E; border-radius:22px; padding:12px 16px; color:#3a2a1a; font-weight:500; }
@media (max-width:600px){ .title{font-size:2rem} .bars .nm{width:104px} }
</style>
""",
    unsafe_allow_html=True,
)

title = "How do you feel today?"
st.markdown(f'<div class="title">{"".join(f"<span>{c}</span>" if c != " " else " " for c in title)}</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub">Write a few words and see what my banner says to you!</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ state
if "text" not in st.session_state:
    st.session_state.text = ""
if "result" not in st.session_state:
    st.session_state.result = None


def use_example(s: str):
    st.session_state.text = s


try:
    model, tokenizer = load_artifacts()
except Exception as e:  # noqa: BLE001
    st.markdown('<div class="err">Oh no, the feelings robot could not wake up. '
                'Check that the Artifacts folder has BiGru_model.keras and tokenizer.pkl.</div>',
                unsafe_allow_html=True)
    st.exception(e)
    st.stop()

# ------------------------------------------------------------------ input
st.text_area("Tell me how you feel", key="text", height=110, max_chars=500,
             placeholder="Today I went to the park and...")

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, key=f"ex_{ex}", on_click=use_example, args=(ex,), use_container_width=True)

go = st.button("Show me my banner! 🎨")

if go:
    text = st.session_state.text.strip()
    if not text:
        st.session_state.result = None
        st.markdown('<div class="err">Type a few words first, then press the button.</div>', unsafe_allow_html=True)
    else:
        probs = predict(text, model, tokenizer)
        top = int(np.argmax(probs))
        st.session_state.result = dict(probs=probs, top=top, quote=random.choice(QUOTES[EMOTIONS[top]]))


def another_quote():
    r = st.session_state.result
    pool = [q for q in QUOTES[EMOTIONS[r["top"]]] if q != r["quote"]]
    r["quote"] = random.choice(pool)


# ------------------------------------------------------------------ result
r = st.session_state.result
if r:
    emotion = EMOTIONS[r["top"]]
    t = THEME[emotion]
    conf = float(r["probs"][r["top"]])
    st.markdown(f'<div class="verdict">{t["emoji"]} {t["label"]}!</div>', unsafe_allow_html=True)
    if conf < 0.45:
        st.markdown('<div class="hint">I am not totally sure, so check the bars below too.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="hint">I am {conf * 100:.0f}% sure.</div>', unsafe_allow_html=True)

    show_scene(scene_html(emotion, r["quote"]))

    c1, c2, c3 = st.columns([1, 2, 1])
    c2.button("Show me another banner 🔄", on_click=another_quote, use_container_width=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    st.markdown(bars_html(r["probs"], r["top"]), unsafe_allow_html=True)
