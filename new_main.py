# ==================== IMPORTS ====================
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import base64
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
import plotly.express as px

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="InsightSphere: An AI-Powered Aspect-Based Sentiment Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💬"
)

# ==================== BACKGROUND AND CSS ====================
def add_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Uncomment below if you have a background image
# add_background("background_image.webp")

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_nlp_models():
    nlp = spacy.load("en_core_web_sm")
    model_path = "./absa_models"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    absa_pipeline = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )
    return nlp, absa_pipeline

nlp, absa_pipeline = load_nlp_models()

# ==================== FUNCTIONS ====================
# ----- Manual Aspect Normalization Map -----
def normalize_aspect(aspect):
    manual_map = {
        # Phone
        "the phone": "phone", "great phone": "phone", "this phone": "phone", "mobile phone": "phone", "smartphone": "phone",
        # Battery
        "the battery": "battery", "battery life": "battery", "the battery life": "battery", "battery backup": "battery",
        "long battery": "battery", "poor battery": "battery",
        # Camera
        "the camera": "camera", "great camera": "camera", "camera quality": "camera", "rear camera": "camera",
        "front camera": "camera", "main camera": "camera", "selfie camera": "camera", "picture quality": "camera",
        "photos": "camera", "images": "camera",
        # Display
        "the display": "display", "the screen": "display", "screen quality": "display", "display quality": "display",
        "touch screen": "display", "screen resolution": "display", "large screen": "display", "amoled screen": "display",
        # Performance
        "fast performance": "performance", "amazing performance": "performance", "great performance": "performance",
        "the performance": "performance", "smooth performance": "performance", "lag free": "performance",
        # Design
        "the design": "design", "sleek design": "design", "a sleek design": "design", "design and build": "design",
        "premium design": "design",
        # Build
        "the build": "build", "build quality": "build", "solid build": "build",
        # Software
        "the software": "software", "user interface": "software", "ui": "software", "os": "software",
        "android version": "software",
        # Fingerprint
        "the fingerprint scanner": "fingerprint scanner", "fingerprint sensor": "fingerprint scanner",
        "fingerprint reader": "fingerprint scanner", "biometric scanner": "fingerprint scanner",
        # Price
        "the price": "price", "price point": "price", "pricing": "price", "value for money": "price", "affordable": "price",
        # Gaming
        "mobile gaming": "gaming", "gaming experience": "gaming", "game performance": "gaming", "gaming phone": "gaming",
        # Sound / Audio
        "sound quality": "audio", "audio output": "audio", "speaker quality": "audio", "the speakers": "audio",
        "loudness": "audio",
        # Heating
        "heating issue": "heating", "overheating": "heating", "gets hot": "heating",
        # Connectivity
        "network signal": "connectivity", "signal strength": "connectivity", "wifi": "connectivity",
        "bluetooth": "connectivity", "call quality": "connectivity",
        # Charging
        "charging speed": "charging", "fast charging": "charging", "charging time": "charging",
        # Usability
        "daily use": "usability", "regular use": "usability",
        # Misc
        "a long time": "duration", "some improvement": "issues"
    }
    return manual_map.get(aspect.lower().strip(), aspect.lower().strip())


def extract_aspects(text):
    doc = nlp(text)

    irrelevant_terms = {
        "i", "it", "you", "he", "she", "we", "they",
        "me", "him", "her", "us", "them",
        "my", "your", "his", "her", "our", "their",
        "mine", "yours", "hers", "ours", "theirs",
        "myself", "yourself", "himself", "herself", "itself",
        "ourselves", "yourselves", "themselves",
        "someone", "anyone", "everyone", "no one", "nobody", "somebody",
        "thing", "something", "anything", "everything",
        "one", "this", "that", "these", "those",
        "nothing", "everything", "the"
    }

    excluded_pos = {
        "PRON", "DET", "AUX", "ADP", "CCONJ", "SCONJ",
        "INTJ", "PART", "NUM", "PUNCT", "SYM", "X"
    }

    aspects = []
    for chunk in doc.noun_chunks:
        root_text = chunk.root.text.lower()
        if (chunk.root.pos_ not in excluded_pos) and (root_text not in irrelevant_terms):
            aspects.append(chunk.text.lower())

    return list(set(aspects))

def analyze_aspect_sentiment(text):
    aspects = extract_aspects(text)
    results = []
    for asp in aspects:
        result = absa_pipeline({"text": text, "text_pair": asp})
        if isinstance(result, list):
            result = result[0]
        results.append({
            "Aspect": asp,
            "Sentiment": result["label"],
            "Confidence": round(result["score"], 3)
        })
    return results


# ==================== SIDEBAR ====================
from PIL import Image
from streamlit_option_menu import option_menu
import streamlit as st

with st.sidebar:
    # ==== Branding ====
    st.markdown("<h2 style='text-align:center; color:#4B8BBE;'>🔍 ABSA Tool</h2>", unsafe_allow_html=True)
    try:
        logo = Image.open("absa_logo.webp")
        st.image(logo, use_container_width =True)
    except:
        st.warning("⚠️ Logo not found!")

    st.markdown("---")

    # ==== Navigation Menu ====
    selected = option_menu(
        menu_title="📂 Main Menu",
        options=["Home", "Manual Analysis", "Batch Analysis", "Visualize Trends", "About Us"],
        icons=["house", "pencil", "file-earmark-spreadsheet", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#F0F2F6"},
            "icon": {"color": "#4B8BBE", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#4B8BBE", "color": "white"},
        }
    )

    # ==== Theme Toggle ====
    st.markdown("### 🎨 Theme Settings")
    theme_choice = st.radio("Choose a Theme", ["Light", "Dark", "Auto"])

    if theme_choice == "Dark":
        st.markdown("""
        <style>
        .stApp { background-color: #1E1E1E; color: #FFFFFF; }
        .stTextInput, .stTextArea, .stSelectbox, .stSlider, .stMultiselect, .stDataFrame { background-color: #2A2A2A; color: #FFFFFF; }
        </style>
        """, unsafe_allow_html=True)
    elif theme_choice == "Light":
        st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; color: #000000; }
        </style>
        """, unsafe_allow_html=True)

    # ==== User Mode Switch ====
    st.markdown("### 👤 User Mode")
    user_mode = st.radio("Select Mode", ["Beginner", "Advanced"])

    # Show tooltips or extra help for beginner
    if user_mode == "Beginner":
        st.info("ℹ️ Beginner mode will display additional guidance and instructions.")

    # ==== Feedback and Resources ====
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [📘 Documentation](https://example.com/docs)
    - [📦 GitHub Repo](https://github.com/your-repo)
    - [📧 Contact Support](mailto:support@example.com)
    """)

    st.markdown("---")

    # ==== Feedback Prompt ====
    st.markdown("### ❤️ Share Your Feedback")
    feedback = st.text_area("Let us know your thoughts!")
    if st.button("📤 Submit Feedback"):
        st.success("Thank you for your feedback!")

    st.markdown("<br><sub style='color:gray;'>v1.0.0 | Built with 💙 by Team ABSA</sub>", unsafe_allow_html=True)


# ==================== PAGE: HOME ====================
# ==================== PAGE: HOME ====================
if selected == "Home":
    st.markdown("""
        <h1 style='text-align: center; color:#4B8BBE;'>InsightSphere: An AI-Powered Aspect-Based Sentiment Analysis Platform</h1>
        <p style='text-align: center; font-size:18px;'>Empowering you with insights from text using advanced NLP models.</p>
        <br><br>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col2:
        try:
            home_img = Image.open("logo1.jpg")
            st.image(home_img, use_container_width=True)
        except:
            pass

    # Features
    st.markdown("""
        <br><br>
        <h3 style='color:#4B8BBE;'>🔍 Features</h3>
        <ul style='font-size:17px;'>
            <li>📍 Sentiment & Emotion Detection</li>
            <li>🔍 Keyword & Aspect Extraction</li>
            <li>📂 CSV Upload for Bulk Analysis</li>
            <li>📈 Interactive WordClouds & Visual Analytics</li>
            <li>🐦 Real-time Tweet Processing (Coming Soon)</li>
        </ul>
    """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("""
    <style>
    .cta-button {
    background-color: #4B8BBE;
    color: white;
    padding: 15px 30px;
    text-decoration: none;
    font-size: 18px;
    border-radius: 8px;
    display: inline-block;
    transition: transform 0.2s;
    }
    .cta-button:hover {
    transform: scale(1.05);
    }
    </style>
    <div style='text-align:center; margin-top:40px;'>
    <a href='#Text Analysis' class='cta-button'>🚀 Try Aspect-Based Analysis Now</a>
    </div>
    """, unsafe_allow_html=True)

    # User Testimonials
    testimonials = [
    {"text": "InsightSphere helped us analyze thousands of customer reviews...", "author": "Priya, Product Manager"},
    {"text": "As a data science student, I used InsightSphere...", "author": "Rohan, MSc AI Student"}
    ]
    testimonial_idx = st.selectbox("View Testimonials", range(len(testimonials)), format_func=lambda i: testimonials[i]["author"])
    st.markdown(f"""
    <div style="background-color:#f9f9f9; padding:20px; border-radius:10px;">
    <p><em>{testimonials[testimonial_idx]["text"]}</em></p>
    <p><strong>– {testimonials[testimonial_idx]["author"]}</strong></p>
    </div>
    """, unsafe_allow_html=True)


# ==================== PAGE: MANUAL ANALYSIS ====================
# ==================== PAGE: MANUAL ANALYSIS ====================
if selected == "Manual Analysis":
    st.markdown("<h2 style='color:#4B8BBE;'>📝 Manual Aspect-Based Sentiment Analysis</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
    This tool allows you to perform <strong>aspect-based sentiment analysis</strong> on individual reviews or comments. 
    Simply type or paste a sentence, paragraph, or product review in the input box below. The system will:
    <ul>
        <li>🔍 Automatically identify the key <strong>aspects</strong> mentioned</li>
        <li>📊 Determine the <strong>sentiment</strong> (Positive, Negative, or Neutral) for each aspect</li>
        <li>📈 Display the results clearly in a structured format</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 💬 Input Example:")
    st.info("Example: *The battery life is excellent, but the screen quality is disappointing.*")

    # Initialize session state
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'results' not in st.session_state:
        st.session_state.results = pd.DataFrame()

    user_input = st.text_area(
        "✍️ Enter your review here:",
        value=st.session_state.user_input,
        height=150,
        key="manual_input",
        placeholder="Type your review or opinion here..."
    )
    st.session_state.user_input = user_input

    st.markdown("✅ Once submitted, you will receive a breakdown of each identified aspect with its sentiment and confidence score.")

    st.markdown("---")
    st.markdown("### 🎯 Sentiment Legend:")
    st.markdown("""
    - 🟢 **Positive**: Favorable opinion detected  
    - 🔴 **Negative**: Unfavorable or critical tone  
    - 🟡 **Neutral**: Objective or mixed feedback  
    """)

    st.markdown("---")

    # ==================== REAL-TIME ASPECT PREVIEW ====================
    if user_input.strip():
        with st.spinner("🧠 Detecting key aspects..."):
            aspects = extract_aspects(user_input)
            if aspects:
                st.markdown("### 🎯 Detected Aspects:")
                st.success(", ".join(aspects))
            else:
                st.info("🤔 No aspects detected. Try a more detailed review.")

    # ==================== ANALYSIS BUTTON ====================
    if st.button("🚀 Analyze Now"):
        if user_input.strip():
            with st.spinner("Analyzing with AI model... please wait"):
                results = analyze_aspect_sentiment(user_input)
                if not results:
                    st.warning("⚠️ No significant aspects found.")
                else:
                    result_df = pd.DataFrame(results)
                    result_df["Sentiment"] = result_df["Sentiment"].str.upper()
                    st.session_state.results = result_df
                    st.success("✅ Analysis Complete! Scroll down to view results.")
                    st.balloons()
        else:
            st.warning("Please enter a review above to start analysis.")

    # ==================== DISPLAY FILTERED RESULTS ====================
    if 'results' in st.session_state and not st.session_state.results.empty:
        st.markdown("## 🎯 Filter Your Sentiment Results")
        unique_sentiments = st.session_state.results["Sentiment"].unique().tolist()
        sentiment_filter = st.multiselect(
            "🎛️ Choose sentiments to view",
            options=unique_sentiments,
            default=unique_sentiments
        )

        filtered_df = st.session_state.results[
            st.session_state.results["Sentiment"].isin(sentiment_filter)
        ]

        if filtered_df.empty:
            st.warning("⚠️ No results match the selected sentiment filters.")
        else:
            st.markdown(f"### 📋 Filtered Results ({len(filtered_df)} aspects):")
            st.dataframe(filtered_df, use_container_width=True)

            # ==================== SUMMARY SENTIMENT COUNTS ====================
            st.subheader("📊 Sentiment Overview")
            sentiment_counts = filtered_df['Sentiment'].value_counts()
            st.write(
                f"""
                <div style="display: flex; gap: 40px;">
                    <div style="padding:10px; background-color:#e0f7fa; border-radius:10px;"><strong>🟢 Positive:</strong> {sentiment_counts.get("POSITIVE", 0)}</div>
                    <div style="padding:10px; background-color:#ffebee; border-radius:10px;"><strong>🔴 Negative:</strong> {sentiment_counts.get("NEGATIVE", 0)}</div>
                    <div style="padding:10px; background-color:#fff9c4; border-radius:10px;"><strong>🟡 Neutral:</strong> {sentiment_counts.get("NEUTRAL", 0)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ==================== MINI BAR CHART ====================
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax)
            ax.set_title("Sentiment Distribution")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # ==================== WORDCLOUD ====================
            st.subheader("☁️ WordCloud of Aspects")
            aspects_text = " ".join(filtered_df["Aspect"])
            if aspects_text:
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(aspects_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("No aspects available for WordCloud after filtering.")

    else:
        st.info("ℹ️ No analysis results to display. Type your review and click **'Analyze Now'**.")


# ==================== PAGE: BATCH ANALYSIS ====================
import logging
import streamlit as st
import pandas as pd

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if selected == "Batch Analysis":
    st.markdown("<h2 style='color:#4B8BBE;'>📂 Batch Aspect-Based Sentiment Analysis</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div style='color: #333; font-size: 16px; line-height: 1.6;'>
        ⭐ <strong>Step 1:</strong> Upload your CSV file containing customer reviews or feedback.<br>
        ⭐ <strong>Step 2:</strong> Select the column that contains the text data.<br>
        ⭐ <strong>Step 3:</strong> Choose batch size and click <strong>'Analyze'</strong> to begin analysis.<br>
        ⭐ <strong>Step 4:</strong> Filter, view and download your results! 📥
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📝 Select Text Column")
        text_column = st.selectbox("🔽 Choose the column containing the review text:", df.columns)

        num_rows = st.slider("🔍 Preview Number of Rows", 1, 10, 5)
        st.dataframe(df[[text_column]].head(num_rows))

        # Initialize session state
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = pd.DataFrame()

        st.markdown("### ⚙️ Batch Settings")
        batch_size = st.slider("🧮 Batch Size per Analysis Cycle", 10, 100, 50)

        if st.button("🚀 Analyze"):
            if text_column not in df.columns:
                st.error("❌ The selected column was not found in the uploaded CSV.")
            else:
                results = []
                total_rows = len(df)
                progress_bar = st.progress(0)
                with st.spinner("⏳ Analyzing reviews, extracting aspects..."):
                    for i in range(0, total_rows, batch_size):
                        batch_df = df[i:i + batch_size]
                        for idx, row in batch_df.iterrows():
                            text = row[text_column]
                            aspects = extract_aspects(text)
                            if len(aspects) > 2:
                                logging.info(f"[Row {idx + 1}] Multiple aspects ({len(aspects)}): {aspects}")
                            for asp in aspects:
                                prediction = absa_pipeline({"text": text, "text_pair": asp})
                                if isinstance(prediction, list):
                                    prediction = prediction[0]
                                normalized_asp = normalize_aspect(asp)
                                results.append({
                                    "Original Text": text,
                                    "Aspect": asp,
                                    "Sentiment": prediction["label"].upper(),
                                    "Normalized Aspect": normalized_asp,
                                    "Confidence": round(prediction["score"], 3)
                                })
                        progress_bar.progress(min((i + batch_size) / total_rows, 1.0))
                    progress_bar.empty()

                result_df = pd.DataFrame(results)
                st.session_state.batch_results = result_df
                st.success("✅ Analysis complete! Results generated below.")

        # Display and filter results
        if not st.session_state.batch_results.empty:
            st.markdown("## 📋 Filter and Explore Results")

            rows_per_page = 50
            total_rows = len(st.session_state.batch_results)
            total_pages = (total_rows + rows_per_page - 1) // rows_per_page

            if total_rows > rows_per_page:
                st.write(f"Showing {total_rows} results across {total_pages} pages.")
                page = st.selectbox("📄 Select Page", list(range(1, total_pages + 1)), key="batch_page")
                start_idx = (page - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, total_rows)
                st.write(f"Page {page}: Rows {start_idx + 1} to {end_idx}")
            else:
                start_idx = 0
                end_idx = total_rows

            st.subheader("🔎 Filter by Sentiment")
            unique_sentiments = st.session_state.batch_results["Sentiment"].unique().tolist()
            sentiment_filter = st.multiselect(
                "🎯 Select Sentiments to Display",
                options=unique_sentiments,
                default=unique_sentiments,
                key="batch_sentiment_filter"
            )

            filtered_df = st.session_state.batch_results[
                st.session_state.batch_results["Sentiment"].isin(sentiment_filter)
            ]

            if filtered_df.empty:
                st.warning("⚠️ No results match the selected sentiment filters.")
            else:
                st.markdown(f"### 📊 Filtered Results ({len(filtered_df)} rows)")
                st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)

                csv = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Filtered Results as CSV",
                    data=csv,
                    file_name="absa_results.csv",
                    mime="text/csv"
                )

                st.subheader("📈 Summary of Aspects and Sentiments")
                summary_df = filtered_df.groupby(["Aspect", "Sentiment"]).size().reset_index(name="Count")
                summary_df = summary_df.sort_values(by="Count", ascending=False)
                if summary_df.empty:
                    st.warning("No summary data available after filtering.")
                else:
                    st.dataframe(summary_df)

        else:
            st.info("ℹ️ No analysis results yet. Upload a CSV and click 'Analyze' to get started.")

# ==================== PAGE: VISUALIZE TRENDS ====================
if selected == "Visualize Trends":
    st.markdown("<h2 style='color:#4B8BBE;'>📊 Aspect-Based Sentiment Visualization Dashboard</h2>", unsafe_allow_html=True)
    
    uploaded_viz = st.file_uploader("📁 Upload your ABSA Results CSV", type=["csv"])

    if uploaded_viz is not None:
        df = pd.read_csv(uploaded_viz)

        if not {"Aspect", "Sentiment"}.issubset(df.columns):
            st.error("❌ CSV must contain 'Aspect' and 'Sentiment' columns.")
        else:
            st.markdown("### 📌 Overview")
            st.write(f"Total Rows: **{len(df)}** | Unique Aspects: **{df['Aspect'].nunique()}**")

            # Optional filters
            with st.expander("🔎 Optional Filters"):
                sentiments = st.multiselect("🎯 Filter by Sentiment", options=df["Sentiment"].unique().tolist(), default=df["Sentiment"].unique().tolist())
                df = df[df["Sentiment"].isin(sentiments)]

                aspect_search = st.text_input("🔍 Search for Specific Aspect (optional)")
                if aspect_search:
                    df = df[df["Aspect"].str.contains(aspect_search, case=False, na=False)]

            # 1. Sentiment Distribution (Pie chart)
            st.markdown("### 📊 Overall Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel"))
            ax1.axis("equal")
            st.pyplot(fig1)

            # 2. Sentiment by Aspect (Interactive bar chart)
            st.markdown("### 📊 Sentiment Breakdown by All Aspects")
            # Grouping data by Aspect and Sentiment
           # grouped_df = df.groupby(["Aspect", "Sentiment"]).size().reset_index(name="Count")
            # Grouping in case of duplicated (Aspect, Sentiment) pairs
            df_grouped = df.groupby(["Aspect", "Sentiment"], as_index=False)["Count"].sum()

            bar_fig = px.bar(
                df_grouped,
                x="Aspect",
                y="Count",
                color="Sentiment",
                barmode="group",
                title="Sentiment Distribution Across All Aspects",
                color_discrete_sequence=px.colors.qualitative.Set2,
                )
            bar_fig.update_layout(
                xaxis_title="Aspect",
                yaxis_title="Count",
                xaxis_tickangle=-45,
                xaxis={'categoryorder': 'total descending'},
                margin=dict(t=60, b=200),
                height=600,
                legend_title="Sentiment",
                )
            st.plotly_chart(bar_fig, use_container_width=True)

            # 3. Word Cloud
            st.markdown("### ☁️ WordCloud of Extracted Aspects")
            aspect_text = " ".join(df["Aspect"].astype(str))
            wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(aspect_text)
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.imshow(wc, interpolation="bilinear")
            ax3.axis("off")
            st.pyplot(fig3)

            # 4. Optional Table View
            with st.expander("📋 View Data Table"):
                st.dataframe(df.reset_index(drop=True))

            st.success("✅ Visualization generated successfully!")

    else:
        st.info("📥 Please upload a CSV file to generate visualizations.")
# ==================== PAGE: ABOUT US ====================
# ==================== MAIN CONTENT ====================
if selected == "About Us":
    st.markdown("<h2 style='text-align: center;'>🤖 About InsightSphere</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: justify; font-size: 16px;">
    <strong>InsightSphere</strong> is a cutting-edge AI-powered platform designed to perform <strong>Aspect-Based Sentiment Analysis (ABSA)</strong> with remarkable precision. It goes beyond basic sentiment analysis by identifying specific <em>aspects</em> or <em>features</em> in textual data (like "battery life" or "customer service") and evaluating the sentiment associated with each. Powered by advanced NLP models and an intuitive interface, InsightSphere brings powerful insights within reach for users from all backgrounds.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 What It Does")
        st.markdown("""
        - Extracts **keywords and meaningful aspects** from feedback, reviews, and text.
        - Evaluates sentiment (Positive, Negative, Neutral) for each **identified aspect**.
        - Supports both **manual text entry** and **bulk analysis** via CSV upload.
        - Generates intuitive visualizations like **pie charts**, **word clouds**, and **bar plots**.
        - Offers reliable and explainable results using **state-of-the-art transformer models**.
        """)

    with col2:
        st.subheader("🎯 Who Can Use InsightSphere?")
        st.markdown("""
        - 🏢 **Businesses & Brands**: To analyze customer reviews and understand product perception.
        - 📊 **Marketing Analysts**: To track public sentiment towards campaigns, features, or competitors.
        - 🧠 **Researchers**: For studying opinions, social trends, and public discourse at a granular level.
        - 🎓 **Students & Educators**: For academic projects and hands-on learning in NLP and machine learning.
        - 👨‍💻 **Developers & Data Scientists**: To integrate ABSA capabilities into custom NLP pipelines.
        - 🛍️ **E-commerce Platforms**: To extract insights from product reviews for vendor/product optimization.
        """)

    st.markdown("---")

    st.subheader("💼 Real-World Applications")
    st.markdown("""
    - 📱 **App Store/Play Store Review Analysis**  
      Understand what users like/dislike across thousands of reviews.
    - 🏨 **Hotel & Restaurant Feedback**  
      Find out if food quality, cleanliness, or service are areas of concern.
    - 🛒 **Product Review Monitoring**  
      Track customer satisfaction across different features (e.g., durability, price, usability).
    - 💬 **Social Media Listening**  
      Monitor conversations around brands, events, or campaigns to respond proactively.
    - 📰 **Media & News Sentiment Mining**  
      Understand public tone around policies, leaders, or current affairs.
    """)

    st.markdown("---")

    st.subheader("🛠️ Technology Stack")
    st.markdown("""
    - **Frontend:** Streamlit, HTML/CSS  
    - **Visualization:** matplotlib, seaborn, wordcloud  
    - **Text Processing:** spaCy, pandas  
    - **Models & NLP:** HuggingFace Transformers, DeBERTa-v3 (ABSA)  
    - **Deployment Ready:** Lightweight, responsive, and scalable for cloud or local use.
    """)

    st.markdown("---")
    
    st.success("🌟 InsightSphere helps you turn feedback into action by combining AI precision with human context.")

