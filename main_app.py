"""
Laboratorio de PLN y LLMs con Groq
==================================
Ejecutar:  streamlit run main_app.py

Pestañas:
  1. Tokenización  -> varios esquemas, IDs de tokens y tokens coloreados
  2. Similitud coseno entre frases (BoW / binario / TF-IDF / n-gramas de caracteres)
  3. Bag of Words  -> vocabulario, matriz documento-término y frecuencias
  4. Catálogo de modelos disponibles en Groq (endpoint /openai/v1/models)
  5. OCR -> extrae texto de imágenes/PDF (Groq visión o Tesseract local) y lo usa
     como prompt para ampliar la respuesta con un LLM
  6. Generación con distintos parámetros (temperatura, top_p, max tokens, seed,
     stop, reasoning_effort) y comparación entre temperaturas o modelos

Fuentes de referencia (verificadas en septiembre de 2026):
  - Modelos Groq:           https://console.groq.com/docs/models
  - Compatibilidad/límites: https://console.groq.com/docs/openai
  - Razonamiento:           https://console.groq.com/docs/reasoning
  - Visión / OCR:           https://console.groq.com/docs/vision
  - API reference:          https://console.groq.com/docs/api-reference
  - tiktoken:               https://github.com/openai/tiktoken
"""

from __future__ import annotations

import base64
import html
import io
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Laboratorio PLN + Groq",
    page_icon="🔤",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Utilidades visuales
# ---------------------------------------------------------------------------
PALETTE = [
    "#FFD6A5", "#CAFFBF", "#9BF6FF", "#BDB2FF", "#FFC6FF", "#FDFFB6",
    "#A0C4FF", "#FFADAD", "#B9FBC0", "#F1C0E8", "#CFBAF0", "#98F5E1",
]

st.markdown(
    """
    <style>
    .tok-wrap {line-height: 2.3; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
               font-size: 0.95rem;}
    .tok {padding: 3px 5px; margin: 1px; border-radius: 5px; color: #1f2328;
          white-space: pre; border: 1px solid rgba(0,0,0,.12);}
    .tok sub {font-size: 0.62rem; color: #57606a; margin-left: 2px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def visible(s: str) -> str:
    """Hace visibles los espacios y saltos de línea dentro de un token."""
    return s.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥")


def render_tokens(tokens: list[tuple[str, int]], show_ids: bool, show_ws: bool) -> str:
    spans = []
    for i, (tok, tid) in enumerate(tokens):
        text = visible(tok) if show_ws else tok
        text = html.escape(text) if text else "∅"
        sub = f"<sub>{tid}</sub>" if show_ids else ""
        color = PALETTE[i % len(PALETTE)]
        spans.append(
            f'<span class="tok" style="background:{color}" title="id={tid}">{text}{sub}</span>'
        )
    return f'<div class="tok-wrap">{"".join(spans)}</div>'


# ---------------------------------------------------------------------------
# Esquemas de tokenización
# ---------------------------------------------------------------------------
def _local_ids(pieces: list[str]) -> list[tuple[str, int]]:
    """IDs de un vocabulario LOCAL construido con el propio texto (orden de aparición)."""
    vocab: dict[str, int] = {}
    return [(p, vocab.setdefault(p, len(vocab))) for p in pieces]


def tok_whitespace(text: str):
    return _local_ids(text.split()), None


def tok_regex(text: str):
    return _local_ids(re.findall(r"\w+|[^\w\s]", text)), None


def tok_chars(text: str):
    return [(c, ord(c)) for c in text], 0x110000  # ID = punto de código Unicode


def tok_bytes(text: str):
    out = []
    for b in text.encode("utf-8"):
        shown = chr(b) if 32 <= b < 127 else f"0x{b:02X}"
        out.append((shown, b))
    return out, 256


@st.cache_resource(show_spinner=False)
def get_tiktoken(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


def make_tiktoken(name: str):
    def _tok(text: str):
        enc = get_tiktoken(name)
        # disallowed_special=() -> textos como "<|endoftext|>" se tratan como texto normal
        ids = enc.encode(text, disallowed_special=())
        pieces = [
            enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids
        ]
        return list(zip(pieces, ids)), enc.n_vocab

    return _tok


@st.cache_resource(show_spinner=False)
def get_hf_tokenizer(repo: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(repo)  # descarga tokenizer.json desde Hugging Face Hub


def make_hf(repo: str):
    def _tok(text: str, add_special: bool = False):
        tk = get_hf_tokenizer(repo)
        enc = tk.encode(text, add_special_tokens=add_special)
        return list(zip(enc.tokens, enc.ids)), tk.get_vocab_size()

    return _tok


SCHEMES = {
    "Espacios en blanco (split)": (
        tok_whitespace,
        "Divide solo por espacios. IDs = vocabulario local (orden de aparición en este texto).",
    ),
    "Palabras + puntuación (regex)": (
        tok_regex,
        r"Regex `\w+|[^\w\s]`: separa palabras y signos. IDs = vocabulario local.",
    ),
    "Caracteres (Unicode)": (
        tok_chars,
        "Un token por carácter. ID = punto de código Unicode (ord).",
    ),
    "Bytes UTF-8": (
        tok_bytes,
        "Un token por byte. ID = valor del byte (0–255). Las tildes y emojis ocupan varios bytes.",
    ),
    "BPE · gpt2 (tiktoken)": (
        make_tiktoken("gpt2"),
        "BPE a nivel de byte de GPT-2.",
    ),
    "BPE · cl100k_base (tiktoken)": (
        make_tiktoken("cl100k_base"),
        "Codificación usada por gpt-4 y gpt-3.5-turbo.",
    ),
    "BPE · o200k_base (tiktoken)": (
        make_tiktoken("o200k_base"),
        "Codificación usada por gpt-4o, serie o y GPT-5.",
    ),
    "BPE · o200k_harmony (tiktoken, GPT-OSS)": (
        make_tiktoken("o200k_harmony"),
        "Codificación de gpt-oss-20b / gpt-oss-120b (los modelos GPT disponibles en Groq).",
    ),
    "WordPiece · bert-base-uncased (HF)": (
        make_hf("bert-base-uncased"),
        "WordPiece de BERT. '##' marca continuación de palabra. Pasa a minúsculas y quita tildes.",
    ),
    "BPE · gpt2 (HF tokenizers)": (
        make_hf("gpt2"),
        "El mismo BPE de GPT-2 visto con HF: 'Ġ' representa un espacio antes del token.",
    ),
    "Unigram SentencePiece · xlm-roberta-base (HF)": (
        make_hf("xlm-roberta-base"),
        "Modelo Unigram multilingüe. '▁' representa un espacio antes del token.",
    ),
}
HF_SCHEMES = {k for k in SCHEMES if "(HF" in k}


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
def groq_client(api_key: str):
    from groq import Groq

    return Groq(api_key=api_key)


def fetch_models(api_key: str) -> list[dict]:
    client = groq_client(api_key)
    models = client.models.list()
    rows = []
    for m in models.data:
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        rows.append(d)
    return sorted(rows, key=lambda r: r.get("id", ""))


NON_CHAT_HINTS = ("whisper", "orpheus", "guard", "tts", "playai")


def is_chat_model(model_id: str) -> bool:
    return not any(h in model_id.lower() for h in NON_CHAT_HINTS)


def is_gpt_oss(model_id: str) -> bool:
    return model_id.startswith("openai/gpt-oss")


# Modelos con controles de razonamiento documentados (docs/reasoning, sept. 2026)
QWEN_REASONING = {"qwen/qwen3.8-27b"}
# Único modelo de visión listado en docs/vision (sept. 2026). Si Groq agrega otros,
# el usuario puede escribir su ID manualmente en la pestaña OCR.
VISION_MODELS = ["qwen/qwen3.8-27b"]


def reasoning_family(model_id: str) -> str | None:
    if model_id in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
        return "gpt-oss"
    if model_id in QWEN_REASONING:
        return "qwen"
    return None


def reasoning_controls(models: list[str], key: str) -> dict:
    """Muestra solo los controles de razonamiento que aplican a los modelos elegidos."""
    fams = {reasoning_family(m) for m in models} - {None}
    out: dict[str, dict] = {}
    if "gpt-oss" in fams:
        c1, c2 = st.columns(2)
        out["gpt-oss"] = {
            "reasoning_effort": c1.selectbox("reasoning_effort · GPT-OSS",
                                             ["low", "medium", "high"], 1, key=f"{key}_oe"),
            "include_reasoning": c2.checkbox("include_reasoning · GPT-OSS", True,
                                             key=f"{key}_oi"),
        }
    if "qwen" in fams:
        c1, c2 = st.columns(2)
        out["qwen"] = {
            "reasoning_effort": c1.selectbox(
                "reasoning_effort · Qwen 3.8", ["none", "default", "low", "medium", "high"],
                1, key=f"{key}_qe",
                help="none desactiva el razonamiento; default no devuelve tokens de razonamiento."),
            # include_reasoning y reasoning_format son mutuamente excluyentes (docs/reasoning)
            "reasoning_format": c2.selectbox("reasoning_format · Qwen 3.8",
                                             ["parsed", "hidden", "raw"], 0, key=f"{key}_qf"),
        }
    return out


def apply_reasoning(params: dict, model: str, rc: dict) -> dict:
    fam = reasoning_family(model)
    if fam in rc:
        params.update(rc[fam])
    return params


def generate(api_key: str, model: str, messages: list[dict], params: dict) -> dict:
    client = groq_client(api_key)
    kwargs = {k: v for k, v in params.items() if v is not None}
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    except Exception as e:  # noqa: BLE001 - mostramos el error de la API tal cual
        return {"error": f"{type(e).__name__}: {e}", "latency": time.perf_counter() - t0}
    latency = time.perf_counter() - t0
    choice = resp.choices[0]
    usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
    content = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning", None)
    think = re.match(r"\s*<think>(.*?)</think>\s*", content, flags=re.S)
    if think and not reasoning:  # reasoning_format="raw" mete el razonamiento en el texto
        reasoning, content = think.group(1).strip(), content[think.end():]
    return {
        "content": content,
        "reasoning": reasoning,
        "finish_reason": choice.finish_reason,
        "usage": {k: v for k, v in usage.items() if isinstance(v, (int, float))},
        "latency": latency,
    }


def show_result(res: dict, title: str):
    st.markdown(f"**{title}**")
    if "error" in res:
        st.error(res["error"])
        return
    if res["content"]:
        st.markdown(res["content"])
    else:
        st.warning("La respuesta llegó vacía.")
    if res["finish_reason"] == "length":
        st.caption(
            "⚠️ finish_reason = length: se alcanzó max_completion_tokens. "
            "En modelos de razonamiento, aumenta el límite o baja reasoning_effort."
        )
    if res.get("reasoning"):
        with st.expander("Ver razonamiento del modelo"):
            st.text(res["reasoning"])
    u = res["usage"]
    st.caption(
        f"finish_reason: {res['finish_reason']} · latencia medida: {res['latency']:.2f} s · "
        f"tokens prompt/completion/total: {u.get('prompt_tokens')}/"
        f"{u.get('completion_tokens')}/{u.get('total_tokens')}"
    )


# ---------------------------------------------------------------------------
# Sidebar: API key (obligatoria para empezar)
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("api_ok", False)
ss.setdefault("api_key", "")
ss.setdefault("models", [])
ss.setdefault("ocr_text", "")
ss.setdefault("tok_text", "La inteligencia artificial en EAFIT: ¡tokenizar 'desafortunadamente' "
              "cuesta más tokens que 'unfortunately'! 🤖")
ss.setdefault("cos_text", "El gato duerme en el sofá\nEl perro duerme en la cama\n"
              "Un felino descansa sobre el sillón\nLa bolsa de valores cayó hoy")
ss.setdefault("bow_text", "El modelo aprende de los datos\nLos datos entrenan el modelo de "
              "lenguaje\nEl lenguaje natural es ambiguo")

with st.sidebar:
    st.header("🔑 Groq API key")
    if not ss.api_ok:
        key_in = st.text_input(
            "Pega tu API key", type="password", placeholder="gsk_...",
            help="Se crea en https://console.groq.com/keys. Solo vive en la sesión del navegador.",
        )
        if st.button("Conectar", type="primary", width="stretch"):
            if not key_in.strip():
                st.error("Ingresa una API key.")
            else:
                with st.spinner("Validando con /models ..."):
                    try:
                        ss.models = fetch_models(key_in.strip())
                        ss.api_key = key_in.strip()
                        ss.api_ok = True
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"No se pudo validar la key: {type(e).__name__}: {e}")
    else:
        st.success(f"Conectado · {len(ss.models)} modelos disponibles")
        if st.button("Refrescar catálogo", width="stretch"):
            try:
                ss.models = fetch_models(ss.api_key)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
        if st.button("Desconectar", width="stretch"):
            ss.api_ok, ss.api_key, ss.models = False, "", []
            st.rerun()
    st.divider()
    st.caption(
        "La key no se guarda en disco: se mantiene en `st.session_state` "
        "y se pierde al cerrar la pestaña."
    )

st.title("🔤 Laboratorio de PLN y LLMs con Groq")

if not ss.api_ok:
    st.info(
        "Para empezar, ingresa tu **API key de Groq** en la barra lateral y presiona "
        "**Conectar**. La app valida la key consultando el catálogo de modelos."
    )
    st.markdown(
        "1. Crea una key en [console.groq.com/keys](https://console.groq.com/keys).\n"
        "2. Pégala en la barra lateral.\n"
        "3. Explora tokenización, similitud coseno, bag of words y generación de texto."
    )
    st.stop()

tab_tok, tab_cos, tab_bow, tab_cat, tab_ocr, tab_gen = st.tabs(
    ["🧩 Tokenización", "📐 Similitud coseno", "👜 Bag of Words",
     "📚 Catálogo de modelos", "🖼️ OCR → Prompt", "✨ Generación"]
)

# ---------------------------------------------------------------------------
# 1. Tokenización
# ---------------------------------------------------------------------------
with tab_tok:
    st.subheader("Esquemas de tokenización")
    text = st.text_area("Texto a tokenizar", key="tok_text", height=100)
    default = [
        "Palabras + puntuación (regex)", "Caracteres (Unicode)",
        "BPE · cl100k_base (tiktoken)", "BPE · o200k_harmony (tiktoken, GPT-OSS)",
        "WordPiece · bert-base-uncased (HF)",
    ]
    chosen = st.multiselect("Esquemas", list(SCHEMES), default=default)
    c1, c2, c3 = st.columns(3)
    show_ids = c1.toggle("Mostrar ID bajo cada token", True)
    show_ws = c2.toggle("Hacer visibles espacios (␣) y saltos (↵)", True)
    add_special = c3.toggle("Tokens especiales en HF ([CLS], <s>…)", False)

    summary = []
    for name in chosen:
        fn, desc = SCHEMES[name]
        st.markdown(f"#### {name}")
        st.caption(desc)
        try:
            with st.spinner("Cargando tokenizador…"):
                tokens, vocab_size = (
                    fn(text, add_special) if name in HF_SCHEMES else fn(text)
                )
        except Exception as e:  # noqa: BLE001
            st.error(
                f"No se pudo cargar este esquema ({type(e).__name__}: {e}). "
                "Los esquemas tiktoken y HF descargan su vocabulario la primera vez: "
                "revisa tu conexión a internet."
            )
            continue

        m1, m2, m3 = st.columns(3)
        m1.metric("Nº de tokens", len(tokens))
        m2.metric("Caracteres / token", f"{len(text) / max(len(tokens), 1):.2f}")
        m3.metric("Tamaño del vocabulario",
                  f"{vocab_size:,}" if vocab_size else "local: "
                  f"{len({t for t, _ in tokens})}")
        st.markdown(render_tokens(tokens, show_ids, show_ws), unsafe_allow_html=True)
        with st.expander("Tabla de tokens e IDs"):
            st.dataframe(
                pd.DataFrame(
                    [{"posición": i, "token": visible(t), "id": tid, "repr": repr(t)}
                     for i, (t, tid) in enumerate(tokens)]
                ),
                width="stretch", hide_index=True,
            )
            st.code(str([tid for _, tid in tokens]), language="python")
        if "tiktoken" in name:
            st.caption(
                "Nota: los tokens BPE de byte pueden contener solo una parte de un carácter "
                "UTF-8 (tildes, emojis); al decodificarlos aislados aparecen como �."
            )
        summary.append({"esquema": name, "tokens": len(tokens)})

    if len(summary) > 1:
        st.markdown("#### Comparación de longitud")
        fig = px.bar(pd.DataFrame(summary), x="tokens", y="esquema", orientation="h",
                     text="tokens")
        fig.update_layout(height=80 + 40 * len(summary), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 2. Similitud coseno
# ---------------------------------------------------------------------------
with tab_cos:
    st.subheader("Similitud coseno entre frases")
    st.latex(r"\cos(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}\cdot\mathbf{b}}"
             r"{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}")
    raw = st.text_area("Una frase por línea", key="cos_text", height=140)
    sents = [s.strip() for s in raw.splitlines() if s.strip()]
    c1, c2, c3 = st.columns(3)
    method = c1.radio(
        "Representación vectorial",
        ["Conteos (BoW)", "Binaria (presencia)", "TF-IDF (palabras)",
         "TF-IDF (n-gramas de caracteres 3–5)"],
    )
    lower = c2.checkbox("Minúsculas", True, key="cos_lower")
    accents = c2.checkbox("Quitar tildes", False, key="cos_acc")
    c3.info(
        "Estas representaciones miden **coincidencia léxica**, no significado: "
        "'gato' y 'felino' son dimensiones distintas y aportan 0 al producto punto."
    )

    if len(sents) < 2:
        st.warning("Escribe al menos dos frases.")
    else:
        common = dict(lowercase=lower, strip_accents="unicode" if accents else None)
        if method == "Conteos (BoW)":
            vec = CountVectorizer(**common)
        elif method == "Binaria (presencia)":
            vec = CountVectorizer(binary=True, **common)
        elif method == "TF-IDF (palabras)":
            vec = TfidfVectorizer(**common)
        else:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common)
        try:
            X = vec.fit_transform(sents)
        except ValueError as e:
            st.error(f"No se pudo vectorizar: {e}")
            X = None
    if len(sents) >= 2 and X is not None:
        S = cosine_similarity(X)
        labels = [f"F{i + 1}" for i in range(len(sents))]

        fig = px.imshow(S, x=labels, y=labels, text_auto=".3f", zmin=0, zmax=1,
                        color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(height=120 + 60 * len(sents))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame({"id": labels, "frase": sents}), hide_index=True,
                     width="stretch")

        st.markdown("#### Cálculo paso a paso para un par")
        p1, p2 = st.columns(2)
        i = p1.selectbox("Frase A", range(len(sents)), format_func=lambda k: labels[k])
        j = p2.selectbox("Frase B", range(len(sents)), index=1,
                         format_func=lambda k: labels[k])
        a = X[i].toarray().ravel()
        b = X[j].toarray().ravel()
        vocab = vec.get_feature_names_out()
        mask = (a != 0) | (b != 0)
        df_pair = pd.DataFrame({"término": vocab[mask], "A": a[mask], "B": b[mask]})
        df_pair["A·B"] = df_pair["A"] * df_pair["B"]
        st.dataframe(df_pair.round(4), hide_index=True, width="stretch")
        dot = float(a @ b)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        cos = dot / (na * nb) if na and nb else 0.0
        st.latex(
            rf"\cos = \frac{{{dot:.4f}}}{{{na:.4f}\times{nb:.4f}}} = {cos:.4f}"
        )
        st.caption(
            "Producto punto = suma de la columna A·B. Norma = raíz de la suma de cuadrados "
            "de cada vector (sobre todo el vocabulario). Con TF-IDF, sklearn ya normaliza "
            "los vectores a norma 1 (norm='l2' por defecto)."
        )

# ---------------------------------------------------------------------------
# 3. Bag of Words
# ---------------------------------------------------------------------------
SPANISH_STOP = [
    "a", "al", "algo", "como", "con", "de", "del", "el", "ella", "en", "entre", "era",
    "es", "esa", "ese", "esta", "este", "fue", "ha", "hay", "la", "las", "le", "lo",
    "los", "mas", "más", "me", "mi", "muy", "no", "nos", "o", "para", "pero", "por",
    "que", "qué", "se", "si", "sin", "sobre", "su", "sus", "también", "te", "tu", "un",
    "una", "uno", "y", "ya", "yo",
]

with tab_bow:
    st.subheader("Bag of Words")
    raw_bow = st.text_area("Un documento por línea", key="bow_text", height=120)
    docs = [d.strip() for d in raw_bow.splitlines() if d.strip()]
    c1, c2, c3 = st.columns(3)
    b_lower = c1.checkbox("Minúsculas", True, key="bow_lower")
    b_acc = c1.checkbox("Quitar tildes", False, key="bow_acc")
    b_bin = c1.checkbox("Binario (presencia 0/1)", False)
    stop_opt = c2.selectbox(
        "Stopwords", ["Ninguna", "Español (lista básica incluida)", "Inglés (sklearn)"]
    )
    ngram = c2.selectbox("n-gramas", ["(1,1) unigramas", "(1,2) uni+bigramas",
                                      "(2,2) bigramas"])
    min_df = c3.number_input("min_df (aparece en ≥ n documentos)", 1, 10, 1)
    top_n = c3.slider("Top términos a graficar", 5, 50, 15)

    stop = {"Ninguna": None, "Inglés (sklearn)": "english",
            "Español (lista básica incluida)": SPANISH_STOP}[stop_opt]
    ng = {"(1,1) unigramas": (1, 1), "(1,2) uni+bigramas": (1, 2),
          "(2,2) bigramas": (2, 2)}[ngram]

    if not docs:
        st.warning("Escribe al menos un documento.")
    else:
        cv = CountVectorizer(lowercase=b_lower, strip_accents="unicode" if b_acc else None,
                             stop_words=stop, ngram_range=ng, binary=b_bin, min_df=min_df)
        try:
            M = cv.fit_transform(docs)
            vocab = cv.get_feature_names_out()
            dtm = pd.DataFrame(M.toarray(), columns=vocab,
                               index=[f"D{k + 1}" for k in range(len(docs))])

            st.markdown(f"**Vocabulario** ({len(vocab)} términos · índice = columna del vector)")
            st.dataframe(
                pd.DataFrame({"índice": range(len(vocab)), "término": vocab}).astype(str).T,
                width="stretch",
            )
            st.markdown("**Matriz documento–término**")
            st.dataframe(dtm.style.background_gradient(cmap="Blues", axis=None),
                         width="stretch")
            st.markdown("**Vector de cada documento**")
            for k, d in enumerate(docs):
                st.code(f"D{k + 1}: {d}\n→ {M[k].toarray().ravel().tolist()}")

            freq = dtm.sum(axis=0).sort_values(ascending=False).head(top_n)
            fig = px.bar(freq[::-1], orientation="h", labels={"value": "frecuencia",
                                                              "index": "término"})
            fig.update_layout(showlegend=False, height=120 + 22 * len(freq))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "El BoW descarta el orden: 'el perro muerde al hombre' y "
                "'el hombre muerde al perro' producen el mismo vector de unigramas."
            )
        except ValueError as e:
            st.error(f"Vocabulario vacío o parámetros inválidos: {e}")

# ---------------------------------------------------------------------------
# 4. Catálogo de modelos
# ---------------------------------------------------------------------------
with tab_cat:
    st.subheader("Catálogo de modelos disponibles para tu key")
    st.caption(
        "Datos obtenidos en vivo de `GET https://api.groq.com/openai/v1/models`. "
        "Las columnas son exactamente los campos que devuelve la API."
    )
    df_models = pd.DataFrame(ss.models)
    if df_models.empty:
        st.warning("La API no devolvió modelos.")
    else:
        if "created" in df_models:
            df_models["created"] = pd.to_datetime(df_models["created"], unit="s",
                                                  errors="coerce").dt.date
        f1, f2 = st.columns([1, 2])
        only_gpt = f1.checkbox("Solo modelos GPT de OpenAI (openai/…)", False)
        query = f2.text_input("Filtrar por texto", "")
        view = df_models
        if only_gpt:
            view = view[view["id"].str.startswith("openai/")]
        if query:
            view = view[view["id"].str.contains(query, case=False, regex=False)]
        st.dataframe(view, width="stretch", hide_index=True)
        st.markdown(
            "Precios, velocidad y estado (producción/preview) se consultan en "
            "[console.groq.com/docs/models](https://console.groq.com/docs/models); "
            "cambian con el tiempo y no se muestran aquí para no inventar cifras."
        )

# ---------------------------------------------------------------------------
# 5. OCR -> Prompt
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def tesseract_langs() -> list[str] | None:
    """Idiomas de Tesseract si el binario está instalado; None si no está disponible."""
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return [l for l in pytesseract.get_languages(config="") if l != "osd"]
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False)
def pdf_page_count(data: bytes) -> int:
    import pypdfium2 as pdfium

    return len(pdfium.PdfDocument(data))


@st.cache_data(show_spinner=False)
def pdf_to_images(data: bytes, first: int, last: int, scale: float) -> list[Image.Image]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    return [pdf[i].render(scale=scale).to_pil() for i in range(first - 1, last)]


def to_data_url(img: Image.Image, max_side: int) -> str:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def conf_color(conf: float) -> str:
    return "#FFADAD" if conf < 60 else "#FDFFB6" if conf < 85 else "#CAFFBF"


def ocr_tesseract(img: Image.Image, lang: str) -> tuple[str, pd.DataFrame]:
    import pytesseract

    text = pytesseract.image_to_string(img, lang=lang)
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DATAFRAME)
    words = data[(data["conf"] >= 0) & data["text"].notna()]
    words = words[words["text"].astype(str).str.strip() != ""][["text", "conf"]]
    return text.strip(), words


OCR_PROMPT = (
    "Transcribe todo el texto visible en la imagen exactamente como aparece, respetando "
    "los saltos de línea y el idioma original. No traduzcas, no resumas y no agregues "
    "comentarios ni texto que no esté en la imagen. Si hay fórmulas, escríbelas en LaTeX. "
    "Si una parte es ilegible, escribe [ilegible]. Devuelve solo el texto."
)

EXPAND_TASKS = {
    "Explicar y ampliar": (
        "A continuación hay un texto extraído por OCR. Explica su contenido en detalle y "
        "amplía cada idea con definiciones, contexto y ejemplos. Distingue claramente lo que "
        "dice el texto de lo que tú agregas como ampliación. Si detectas posibles errores de "
        "OCR, señálalos en vez de adivinar."
    ),
    "Resumir": (
        "Resume el siguiente texto extraído por OCR en 5–8 líneas, sin agregar información "
        "que no esté en el texto."
    ),
    "Preguntas de estudio": (
        "Con base únicamente en el siguiente texto extraído por OCR, genera 5 preguntas de "
        "estudio con su respuesta. Si el texto no alcanza para responder algo, dilo."
    ),
    "Corregir errores de OCR": (
        "Corrige solo errores evidentes de reconocimiento (letras cambiadas, palabras "
        "partidas, tildes) en el siguiente texto extraído por OCR, sin cambiar su contenido. "
        "Luego lista los cambios que hiciste."
    ),
    "Personalizada": "",
}


def _send_ocr_to(target: str):
    lines = [l.strip() for l in ss.ocr_text.splitlines() if l.strip()]
    if target == "tok_text":
        ss.tok_text = ss.ocr_text
    else:  # cos_text / bow_text: una línea no vacía = una frase/documento
        ss[target] = "\n".join(lines)


with tab_ocr:
    st.subheader("OCR → texto → prompt ampliado")
    st.caption(
        "Paso 1: extrae texto de imágenes o PDF. Paso 2: revisa/corrige el texto. "
        "Paso 3: úsalo como prompt para que un LLM amplíe la respuesta."
    )

    # ---- Paso 1: entrada
    src = st.radio("Fuente", ["Subir imágenes", "Subir PDF", "Cámara"], horizontal=True)
    images: list[tuple[str, Image.Image]] = []
    if src == "Subir imágenes":
        files = st.file_uploader("Imágenes (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"],
                                 accept_multiple_files=True)
        images = [(f.name, Image.open(f)) for f in files or []]
    elif src == "Subir PDF":
        pdf_file = st.file_uploader("PDF", type=["pdf"])
        if pdf_file:
            data = pdf_file.getvalue()
            n = pdf_page_count(data)
            p1, p2 = st.columns(2)
            first, last = (p1.slider("Páginas", 1, n, (1, min(n, 3))) if n > 1 else (1, 1))
            scale = p2.slider("Escala de render (2 ≈ 144 dpi)", 1.0, 4.0, 2.0, 0.5)
            images = [(f"página {first + k}", im)
                      for k, im in enumerate(pdf_to_images(data, first, last, scale))]
    else:
        shot = st.camera_input("Toma una foto del texto")
        if shot:
            images = [("cámara", Image.open(shot))]

    if images:
        st.image([im for _, im in images], caption=[n for n, _ in images], width=220)

    # ---- Motor de OCR
    langs = tesseract_langs()
    engines = ["Groq visión (modelo multimodal)"] + (
        ["Tesseract local (OCR clásico)"] if langs is not None else [])
    engine = st.radio("Motor de OCR", engines, horizontal=True)
    if langs is None:
        st.caption(
            "Tesseract no está instalado en este equipo, por eso solo aparece Groq. "
            "Instalación: https://tesseract-ocr.github.io/ (en Ubuntu: "
            "`sudo apt install tesseract-ocr tesseract-ocr-spa`)."
        )

    if engine.startswith("Groq"):
        available = {m["id"] for m in ss.models}
        vis = [m for m in VISION_MODELS if m in available]
        o1, o2 = st.columns([2, 1])
        if vis:
            vmodel = o1.selectbox("Modelo de visión", vis + ["Otro (escribir ID)"])
        else:
            o1.warning("Ningún modelo de visión conocido aparece en tu catálogo.")
            vmodel = "Otro (escribir ID)"
        if vmodel == "Otro (escribir ID)":
            vmodel = o1.text_input("ID del modelo de visión", VISION_MODELS[0])
        max_side = o2.select_slider("Lado máximo de la imagen (px)",
                                    [768, 1024, 1536, 2048, 3072], 2048)
        ocr_prompt = st.text_area("Instrucción de OCR", OCR_PROMPT, height=110)
        st.caption(
            "Según docs/vision de Groq: máx. 3 imágenes por petición, cada imagen cuenta como "
            "2048 tokens de entrada y una petición con imagen por URL no puede pasar de 20 MB. "
            "Aquí se envía una imagen por petición (base64, JPEG)."
        )
    else:
        default_lang = [l for l in ("spa", "eng") if l in langs] or langs[:1]
        t_langs = st.multiselect("Idiomas de Tesseract", langs, default=default_lang)

    if st.button("Extraer texto", type="primary", disabled=not images):
        parts, all_words = [], []
        prog = st.progress(0.0)
        for k, (name, img) in enumerate(images):
            header = f"--- {name} ---\n" if len(images) > 1 else ""
            if engine.startswith("Groq"):
                url = to_data_url(img, max_side)
                if len(url) > 20 * 1024 * 1024:
                    st.error(f"{name}: la imagen codificada supera 20 MB; reduce el lado máximo.")
                    continue
                msgs = [{"role": "user", "content": [
                    {"type": "text", "text": ocr_prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ]}]
                params = {"temperature": 0.1, "max_completion_tokens": 4096}
                if vmodel in QWEN_REASONING:
                    params["reasoning_effort"] = "none"  # OCR no necesita razonamiento
                res = generate(ss.api_key, vmodel, msgs, params)
                if "error" in res:
                    st.error(f"{name}: {res['error']}")
                    continue
                parts.append(header + res["content"].strip())
                st.caption(f"{name}: {res['usage'].get('prompt_tokens')} tokens de entrada, "
                           f"{res['usage'].get('completion_tokens')} de salida, "
                           f"{res['latency']:.1f} s")
            else:
                text_k, words = ocr_tesseract(img, "+".join(t_langs) or "eng")
                parts.append(header + text_k)
                all_words.append((name, words))
            prog.progress((k + 1) / len(images))
        ss.ocr_text = "\n\n".join(parts)
        ss.ocr_words = all_words

    # Confianza por palabra (solo Tesseract)
    for name, words in ss.get("ocr_words", []):
        if words.empty:
            continue
        with st.expander(f"Confianza por palabra · {name} "
                         f"(media {words['conf'].mean():.1f}/100)"):
            spans = "".join(
                f'<span class="tok" style="background:{conf_color(c)}" '
                f'title="conf={c:.1f}">{html.escape(str(t))}<sub>{c:.0f}</sub></span>'
                for t, c in zip(words["text"], words["conf"])
            )
            st.markdown(f'<div class="tok-wrap">{spans}</div>', unsafe_allow_html=True)
            st.caption("Rojo < 60, amarillo < 85, verde ≥ 85 (confianza reportada por Tesseract).")

    # ---- Paso 2: texto editable
    st.markdown("#### Texto extraído (editable)")
    st.text_area("Corrige aquí los errores de OCR antes de usarlo", key="ocr_text", height=220)
    if ss.ocr_text.strip():
        n_tokens = "—"
        try:
            n_tokens = len(get_tiktoken("o200k_harmony").encode(ss.ocr_text,
                                                                 disallowed_special=()))
        except Exception:  # noqa: BLE001
            pass
        m1, m2, m3 = st.columns(3)
        m1.metric("Caracteres", len(ss.ocr_text))
        m2.metric("Palabras", len(ss.ocr_text.split()))
        m3.metric("Tokens (o200k_harmony, GPT-OSS)", n_tokens)
        b1, b2, b3, b4 = st.columns(4)
        b1.button("→ Tokenización", on_click=_send_ocr_to, args=("tok_text",), width="stretch")
        b2.button("→ Similitud coseno", on_click=_send_ocr_to, args=("cos_text",),
                  width="stretch")
        b3.button("→ Bag of Words", on_click=_send_ocr_to, args=("bow_text",), width="stretch")
        b4.download_button("Descargar .txt", ss.ocr_text, "texto_ocr.txt", width="stretch")

        # ---- Paso 3: ampliar con un LLM
        st.markdown("#### Ampliar la respuesta usando el texto como prompt")
        chat_ids = [m["id"] for m in ss.models if is_chat_model(m["id"])]
        e1, e2 = st.columns([2, 1])
        emodel = e1.selectbox(
            "Modelo", chat_ids,
            index=chat_ids.index("openai/gpt-oss-20b") if "openai/gpt-oss-20b" in chat_ids else 0,
            key="ocr_model",
        )
        task = e2.selectbox("Tarea", list(EXPAND_TASKS))
        instr = st.text_area("Instrucción", EXPAND_TASKS[task], key=f"ocr_instr_{task}",
                             height=100)
        g1, g2 = st.columns(2)
        etemp = g1.slider("temperature", 0.0, 2.0, 0.6, 0.05, key="ocr_temp")
        emax = g2.number_input("max_completion_tokens", 256, 65536, 4096, 256, key="ocr_max")
        erc = reasoning_controls([emodel], key="ocr")
        final_prompt = f"{instr.strip()}\n\n<texto_ocr>\n{ss.ocr_text.strip()}\n</texto_ocr>"
        with st.expander("Ver el prompt final que se enviará"):
            st.code(final_prompt, language="markdown")
        if st.button("Ampliar respuesta", type="primary", disabled=not instr.strip()):
            eparams = apply_reasoning({"temperature": etemp, "max_completion_tokens": int(emax)},
                                      emodel, erc)
            with st.spinner(f"Generando con {emodel}…"):
                res = generate(ss.api_key, emodel,
                               [{"role": "user", "content": final_prompt}], eparams)
            show_result(res, f"{emodel} · {task}")

# ---------------------------------------------------------------------------
# 6. Generación
# ---------------------------------------------------------------------------
with tab_gen:
    st.subheader("Generación de respuestas con distintos parámetros")
    all_ids = [m["id"] for m in ss.models]
    show_all = st.checkbox("Mostrar también modelos no conversacionales", False)
    ids = all_ids if show_all else [m for m in all_ids if is_chat_model(m)]
    if not ids:
        st.warning("No hay modelos conversacionales disponibles para esta key.")
        st.stop()
    default_idx = ids.index("openai/gpt-oss-20b") if "openai/gpt-oss-20b" in ids else 0

    mode = st.radio("Modo", ["Una respuesta", "Barrido de temperatura", "Comparar modelos"],
                    horizontal=True)
    if mode == "Comparar modelos":
        gpt_defaults = [m for m in ids if is_gpt_oss(m)][:2] or ids[:2]
        models_sel = st.multiselect("Modelos", ids, default=gpt_defaults)
    else:
        models_sel = [st.selectbox("Modelo", ids, index=default_idx)]

    system = st.text_area("System prompt (opcional)",
                          "Eres un asistente conciso que responde en español.", height=70)
    prompt = st.text_area("Prompt", "Escribe una frase creativa sobre la tokenización.",
                          height=100)

    st.markdown("**Parámetros de muestreo**")
    c1, c2, c3, c4 = st.columns(4)
    if mode == "Barrido de temperatura":
        temps_txt = c1.text_input("Temperaturas (coma)", "0.0, 0.7, 1.4")
        try:
            temps = [float(t) for t in temps_txt.split(",") if t.strip()]
        except ValueError:
            temps = []
            c1.error("Formato inválido")
    else:
        temps = [c1.slider("temperature", 0.0, 2.0, 0.7, 0.05,
                           help="Groq acepta (0, 2]. Un 0 se convierte en 1e-8.")]
    top_p = c2.slider("top_p", 0.0, 1.0, 1.0, 0.05,
                      help="Muestreo de núcleo: solo tokens dentro de esa masa de probabilidad.")
    max_tok = c3.number_input("max_completion_tokens", 16, 65536, 1024, 16)
    seed_on = c4.checkbox("Usar seed", False)
    seed = c4.number_input("seed", 0, 2**31 - 1, 42, disabled=not seed_on)
    stop_txt = st.text_input("Secuencias de parada (stop), separadas por '|', máx. 4", "")

    rc = reasoning_controls(models_sel, key="gen")
    use_ocr = st.checkbox(
        "Adjuntar el texto extraído por OCR como contexto", False,
        disabled=not ss.ocr_text.strip(),
        help="Se habilita cuando hay texto en la pestaña OCR → Prompt.",
    )

    base = {
        "top_p": top_p,
        "max_completion_tokens": int(max_tok),
        "seed": int(seed) if seed_on else None,
        "stop": [s for s in stop_txt.split("|") if s][:4] or None,
    }

    def params_for(model: str, temperature: float) -> dict:
        return apply_reasoning(dict(base, temperature=temperature), model, rc)

    user_content = prompt + (f"\n\n<texto_ocr>\n{ss.ocr_text}\n</texto_ocr>" if use_ocr else "")
    messages = ([{"role": "system", "content": system}] if system.strip() else []) + [
        {"role": "user", "content": user_content}
    ]

    if st.button("Generar", type="primary", disabled=not prompt.strip() or not models_sel):
        jobs = (
            [(models_sel[0], t) for t in temps]
            if mode == "Barrido de temperatura"
            else [(m, temps[0]) for m in models_sel]
        )
        cols = st.columns(min(len(jobs), 3)) if jobs else []
        for k, (m, t) in enumerate(jobs):
            with cols[k % len(cols)]:
                with st.spinner(f"{m} · T={t}"):
                    res = generate(ss.api_key, m, messages, params_for(m, t))
                show_result(res, f"{m} · temperature={t}")
        with st.expander("Parámetros enviados (el último trabajo)"):
            st.json({"model": jobs[-1][0], **{k: v for k, v in
                     params_for(*jobs[-1]).items() if v is not None}})

    with st.expander("¿Y el learning rate? (no es un parámetro de generación)"):
        st.markdown(
            "El *learning rate* es un hiperparámetro de **entrenamiento**: controla el "
            "tamaño del paso en el descenso de gradiente al ajustar los pesos. Al generar "
            "texto con un modelo ya entrenado vía API no se actualizan pesos, por eso Groq "
            "no expone ese parámetro en chat completions. Lo que sí controla la generación "
            "son parámetros de **muestreo** (temperature, top_p) y de **longitud/parada**.\n\n"
            "Ilustración de juguete (no es un LLM): minimizar $f(w)=(w-3)^2$ con "
            r"$w_{t+1}=w_t-\eta\,f'(w_t)=w_t-2\eta(w_t-3)$."
        )
        g1, g2, g3 = st.columns(3)
        lr = g1.slider("learning rate η", 0.01, 1.2, 0.1, 0.01)
        w0 = g2.number_input("w₀", -10.0, 10.0, -4.0)
        steps = g3.slider("Iteraciones", 1, 50, 20)
        ws = [w0]
        for _ in range(steps):
            ws.append(ws[-1] - lr * 2 * (ws[-1] - 3))
        grid = np.linspace(-10, 16, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=(grid - 3) ** 2, name="f(w)"))
        fig.add_trace(go.Scatter(x=ws, y=[(w - 3) ** 2 for w in ws], mode="lines+markers",
                                 name="trayectoria"))
        fig.update_layout(height=320, xaxis_title="w", yaxis_title="f(w)",
                          yaxis_range=[0, max(60, (w0 - 3) ** 2 * 1.1)])
        st.plotly_chart(fig, width="stretch")
        factor = abs(1 - 2 * lr)
        st.caption(
            f"Cada paso multiplica el error (w−3) por (1−2η) = {1 - 2 * lr:.2f}. "
            f"|1−2η| = {factor:.2f} → "
            + ("converge." if factor < 1 else "no converge (oscila o diverge).")
        )

    with st.expander("Parámetros que no se incluyen y por qué"):
        st.markdown(
            "- `logprobs`, `logit_bias`, `top_logprobs`: según la guía de compatibilidad "
            "de Groq devuelven error 400; `n` debe ser 1 "
            "([docs](https://console.groq.com/docs/openai)).\n"
            "- `presence_penalty` / `frequency_penalty`: la API reference de Groq indica que "
            "aún no los soporta ningún modelo "
            "([api-reference](https://console.groq.com/docs/api-reference)).\n"
            "- `reasoning_format` no aplica a GPT-OSS (allí se usa `include_reasoning`); "
            "en Qwen 3.8 sí aplica y es excluyente con `include_reasoning` "
            "([reasoning](https://console.groq.com/docs/reasoning))."
        )
