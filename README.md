# How Do You Feel Today? 🌈

A kid-friendly emotion detector. You type a sentence, a deep learning model figures out the emotion, and a cartoon child holds up a banner with a kind quote to match your mood.

Built with TensorFlow/Keras (Bidirectional GRU) and Streamlit.

---

## What it does

1. You write how you feel (for example: *"I feel scared of the dark"*).
2. The text is turned into numbers with the same tokenizer used during training.
3. A **Bidirectional GRU** model gives a probability for each of 6 emotions.
4. The app shows the top emotion, how sure the model is, and a colorful bar for every emotion.
5. An animated kid holds a banner with a short, encouraging quote for that emotion. Press **"Show me another banner"** to get a different quote.

## How to use the app (step by step)

Nothing happens while you are typing. The app only gives a result **after you press the yellow button**.

1. **Type your sentence** in the big white box under "Tell me how you feel". Start it with **"I feel..."** (for example: *I feel so happy today*).
2. **Press the yellow button that says "Show me my banner! 🎨"** (just under the box). This is the main button, and it is easy to miss. Without this click, no result will appear. On a computer you can also press `Ctrl + Enter` instead.
3. **Wait a second.** The first time after the app wakes up it can take a little longer.
4. **See your result:** the emotion name, how sure the model is, the kid holding the banner with a quote, and the bars below.
5. **Want a different quote?** Press **"Show me another banner 🔄"**. The emotion stays the same and only the quote changes.
6. **Want to try again?** Change the text in the box and press the yellow **"Show me my banner!"** button again.

**Shortcut:** under the box there are three white buttons ("I feel so happy today", "I feel sad and alone", "I feel scared of the dark"). Clicking one of these fills the box **and** shows the result right away, so you don't need to press the yellow button for those.

> **Common mistake:** typing a sentence and waiting. The result only shows up after pressing the yellow **"Show me my banner!"** button.

| Emotion  | What the kid looks like | Example banner |
|----------|-------------------------|----------------|
| Sadness  | Blue shirt, gentle frown, tiny tear | "It's okay to cry. Tears help your heart feel lighter." |
| Joy      | Orange shirt, spiky hair, big smile | "Your smile is a little piece of sunshine. Share it!" |
| Love     | Pink shirt, pigtails, rosy cheeks | "A kind word is like a warm hug." |
| Anger    | Red shirt, blue cap, frowning brows | "Count to ten. Your calm is stronger than your grumpy." |
| Fear     | Purple shirt, curly hair, wide eyes | "Take one tiny step. Tiny steps are brave too!" |
| Surprise | Teal shirt, hair bun, "O" mouth | "Wow! Life is full of fun surprises!" |

Each emotion has 5 original, kid-friendly quotes. The UI has a kindergarten look: dotted paper background, crayon-outlined buttons, rainbow title, and the Fredoka font. Animations switch off automatically for people who prefer reduced motion.

---

## The model

- **Dataset:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (English tweets labeled with 6 emotions).
- **Labels (in order):** `sadness, joy, love, anger, fear, surprise`
- **Preprocessing:** Keras `Tokenizer` (10,000 words, `<unk>` for unknown words), sequences padded or truncated to **50 tokens** (`padding="post"`, `truncating="post"`). No extra text cleaning.
- **Architecture:** Embedding (300) -> Bidirectional GRU (128) -> Dropout -> Bidirectional GRU (64) -> Dropout -> Dense (6, softmax).
- **Training:** Adam, sparse categorical cross-entropy, class weights for imbalance, early stopping.

### Models compared in the notebook

| Model  | Test accuracy |
|--------|---------------|
| SimpleRNN | ~18% |
| LSTM | ~34.8% |
| **Bidirectional GRU (used in the app)** | **~92.4%** |

---

## Limitations (please read)

### 1. The model depends heavily on the words "I feel"
The training dataset is made of tweets where almost every sentence follows the pattern **"i feel ..."** or **"i am feeling ..."**. Because of that, the words *i* and *feel* appear extremely often and the model has learned to treat that phrasing as the normal shape of an input. Sentences that don't use it are "out of distribution", and the model often guesses wrong.

Real examples from testing:

| Sentence | Prediction | Comment |
|----------|-----------|---------|
| I feel so happy and excited about my birthday party | Joy (100%) | Correct |
| I feel so scared of the dark | Fear (100%) | Correct |
| I won a prize today! | Fear (54%) | Wrong, no "I feel" |
| I miss my best friend | Fear (43%) | Wrong, no "I feel" |
| I can't believe how happy I am right now, this is amazing! | Surprise (71%) | Should be Joy |

**Tip for users:** start your sentence with *"I feel..."* for the best results. The app's placeholder and example buttons are written this way on purpose.

### 2. Other things to know
- **English only.** Other languages are not understood.
- **Only 6 emotions.** Feelings like "tired", "bored", or "proud" get squeezed into the closest of the six.
- **Short, tweet-style text works best.** Anything beyond 50 tokens is cut off.
- **Vocabulary is limited to 10,000 words.** Rare words and slang are treated as unknown.
- **Mixed feelings:** the model picks one winner. Check the bars to see the runner-up. If the top score is under 45%, the app says it isn't totally sure.
- **It's for fun and learning.** It is not a mental health tool and should not be used to diagnose or judge how someone feels.

### Ideas to improve it
- Train on more varied data (not just "I feel" tweets), or mix in other emotion datasets and child-written sentences.
- Remove or randomly drop "i feel" / "i am feeling" during training so the model learns the emotion words themselves.
- Fine-tune a pretrained transformer (for example DistilBERT) for better handling of everyday sentences.

---

## Project structure

```
emotion_app/
├── app.py                  # Streamlit app (UI, prediction, kid drawings)
├── requirements.txt        # Pinned dependencies
├── .python-version         # Python 3.12
├── .streamlit/config.toml  # Theme and server settings
├── Artifacts/
│   ├── BiGru_model.keras   # Trained model (~42 MB)
│   └── tokenizer.pkl       # Fitted Keras tokenizer
└── README.md
```

The training code lives in the notebook `deep_learning_on_the_go.ipynb`.

---

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

Tested with Python 3.12, `streamlit==1.66.1`, `tensorflow-cpu==2.21.0`, `keras==3.13.2`. The model was saved with Keras 3.13.2, so keep that version.

---

## Deploy on Streamlit Community Cloud

1. Push this whole folder to a GitHub repository (keep `Artifacts/`; at ~42 MB it fits without Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io) and choose **New app**.
3. Pick the repo, the branch, and `app.py` as the main file.
4. Under **Advanced settings** choose **Python 3.12**, then click **Deploy**.

The first build can take several minutes because TensorFlow is large, and the first prediction after the app wakes up is a little slow.

---

## Live demo

🔗 **Deployed app:** https://dsub3pchi7ioj5jbdtc9z9.streamlit.app/

---

## Tech stack

Python, TensorFlow/Keras, NumPy, Streamlit, SVG and CSS animation (the kid and banner are drawn in code, no image files needed).s
