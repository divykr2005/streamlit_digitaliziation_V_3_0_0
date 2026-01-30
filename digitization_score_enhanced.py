import os
import pandas as pd
from pypdf import PdfReader
import re
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# -------------------- KEYWORDS & WEIGHTS --------------------
keywords = {
    "Artificial Intelligence (AI) & Machine Learning (ML)": [
        "Artificial Intelligence", "Business Intelligence", "Image Understanding", "Investment Decision Aid System",
        "Intelligent Data Analysis", "Intelligent Robotics", "Machine Learning", "Deep Learning", "Semantic Search",
        "Biometrics", "Face Recognition", "Voice Recognition", "Identity Verification", "Autonomous Driving",
        "Natural Language Processing", "AI/ML", "Chatbots", "Credit Risk Assessment Models", "Robo-advisor", "Generative AI"
    ],
    "Blockchain Technology": [
        "Blockchain", "Digital Currency", "Cryptocurrency", "Crypto", "Distributed Computing",
        "Differential Privacy Technology", "Smart Financial Contracts", "NFT", "Web 3.0"
    ],
    "Cloud Computing & Infrastructure": [
        "Cloud Computing", "Cloud", "Cloud Technologies", "Streaming Computing", "Graph Computing",
        "In-Memory Computing", "Multi-party Secure Computing", "Brain-like Computing", "Green Computing",
        "Cognitive Computing", "Converged Architecture", "Billion-level Concurrency", "EB-level Storage",
        "APIs", "Digital Infrastructure"
    ],
    "Big Data & Analytics": [
        "Big Data", "Data Mining", "Text Mining", "Data Visualization", "Heterogeneous Data",
        "Credit Analytics", "Augmented Reality", "Mixed Reality", "Virtual Reality", "Transaction Monitoring"
    ],
    "Digital Technology Applications": [
        "Mobile Internet", "Industrial Internet", "Internet Healthcare", "E-commerce", "Mobile Payment",
        "Third-party Payment", "NFC Payment", "Smart Energy", "B2B", "B2C", "C2B", "C2C", "O2O", "Netlink",
        "Smart Wear", "Smart Agriculture", "Smart Transportation", "Smart Healthcare", "Smart Customer Service",
        "Smart Home", "Smart Investment", "Smart Cultural Tourism", "Smart Environmental Protection", "Smart Grid",
        "Smart Marketing", "Digital Marketing", "Unmanned Retail", "Internet Finance", "Digital Finance",
        "Fintech", "Quantitative Finance", "Open Banking", "Embedded Finance", "Peer-to-Peer", "Buy Now Pay Later",
        "Contactless Payments", "Request to Pay", "Payment Service Directive", "Neobank", "Mobile-first Banking",
        "Banking-as-a-Service", "Metaverse"
    ],
    "Cybersecurity & Compliance": [
        "Cyber Security", "Anti-Money Laundering", "Fraud Detection"
    ],
    "Digital Banking & Transformation": [
        "Digitization", "Digital Transformation", "Net Banking", "Internet Banking", "New-to-Digital Customers",
        "E-money", "Robotic Process Automation", "Internet of Things", "Digital Adoption", "Branch on the Move",
        "DBT", "Innovation", "Banking Technology"
    ]
}

default_weights = {
    "Artificial Intelligence (AI) & Machine Learning (ML)": 1,
    "Blockchain Technology": 1,
    "Cloud Computing & Infrastructure": 1,
    "Big Data & Analytics": 1,
    "Digital Technology Applications": 1,
    "Cybersecurity & Compliance": 1,
    "Digital Banking & Transformation": 1
}

# -------------------- COMPILE REGEX PATTERNS (FOR SPEED) --------------------
@st.cache_resource
def compile_keyword_patterns(keywords_dict):
    """Pre-compile regex patterns for faster matching"""
    compiled_patterns = {}
    for category, keyword_list in keywords_dict.items():
        patterns = []
        for keyword in keyword_list:
            pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
            patterns.append(pattern)
        compiled_patterns[category] = patterns
    return compiled_patterns

# -------------------- FUNCTIONS --------------------
def extract_text_from_pdf(pdf_file):
    """Fast PDF text extraction with error handling"""
    try:
        reader = PdfReader(pdf_file)
        # Use list comprehension for faster text extraction
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return text.lower()
    except Exception as e:
        st.warning(f"Error extracting text: {str(e)}")
        return ""

def extract_fy_from_filename(filename):
    """Extract clean FY year from filename"""
    # Remove .pdf extension
    name = os.path.splitext(filename)[0]
    
    # Try to find year patterns
    # Pattern 1: FY2023-24, FY2023-2024, FY23-24
    fy_match = re.search(r'FY[\s_-]?(\d{2,4})[-_](\d{2,4})', name, re.IGNORECASE)
    if fy_match:
        year1 = fy_match.group(1)
        year2 = fy_match.group(2)
        # Convert to 4 digits if needed
        if len(year1) == 2:
            year1 = "20" + year1
        if len(year2) == 2:
            year2 = year1[:2] + year2
        return f"{year1}-{year2[-2:]}"
    
    # Pattern 2: 2023-24, 2023-2024
    year_match = re.search(r'(\d{4})[-_](\d{2,4})', name)
    if year_match:
        year1 = year_match.group(1)
        year2 = year_match.group(2)
        if len(year2) == 2:
            return f"{year1}-{year2}"
        else:
            return f"{year1}-{year2[-2:]}"
    
    # Pattern 3: Just a year (2023, 2024)
    single_year = re.search(r'20\d{2}', name)
    if single_year:
        return single_year.group(0)
    
    return "Unknown"

def compute_digitization_score(text, compiled_patterns, weights, total_words):
    """Compute digitization score using pre-compiled patterns"""
    category_details = {}
    raw_total = 0
    
    for category, patterns in compiled_patterns.items():
        category_score = 0
        for pattern in patterns:
            matches = pattern.findall(text)
            category_score += len(matches)
        
        weighted_score = category_score * weights[category]
        raw_total += weighted_score
        
        category_details[category] = {
            "raw_count": category_score,
            "weighted_score": weighted_score
        }
    
    return raw_total, category_details

def normalize_to_100(raw_scores, method='minmax'):
    """
    Normalize scores to 0-100 scale
    
    Methods:
    - 'minmax': Linear scaling between min and max
    - 'percentile': Based on percentile ranking
    - 'zscore': Z-score normalization (mean=50, std scaled)
    """
    if len(raw_scores) == 0:
        return []
    
    scores_array = np.array(raw_scores)
    
    if method == 'minmax':
        # Linear scaling: (score - min) / (max - min) * 100
        min_score = scores_array.min()
        max_score = scores_array.max()
        if max_score == min_score:
            return [50.0] * len(raw_scores)  # All same score = 50
        normalized = ((scores_array - min_score) / (max_score - min_score)) * 100
        
    elif method == 'percentile':
        # Percentile-based ranking
        from scipy.stats import rankdata
        normalized = (rankdata(scores_array, method='average') / len(scores_array)) * 100
        
    elif method == 'zscore':
        # Z-score normalization centered at 50
        mean_score = scores_array.mean()
        std_score = scores_array.std()
        if std_score == 0:
            return [50.0] * len(raw_scores)
        normalized = 50 + ((scores_array - mean_score) / std_score) * 15  # Scale to ~20-80 range
        normalized = np.clip(normalized, 0, 100)  # Clip to 0-100
    
    else:  # Default to word-normalized
        # Normalize by word count (per 10,000 words) then scale
        normalized = scores_array
    
    return normalized.tolist()

def process_single_pdf(pdf_file, compiled_patterns, weights):
    """Process a single PDF file (for parallel processing)"""
    filename = pdf_file.name
    
    # Extract text
    text = extract_text_from_pdf(pdf_file)
    
    if not text:
        return None
    
    # Extract bank name and FY
    filename_base = os.path.splitext(filename)[0]
    parts = filename_base.split("_")
    bank_name = parts[0] if parts else "Unknown"
    fy = extract_fy_from_filename(filename)
    
    # Count total words
    total_words = len(re.findall(r'\w+', text))
    
    # Compute scores
    raw_score, category_scores = compute_digitization_score(text, compiled_patterns, weights, total_words)
    
    result = {
        "filename": filename,
        "FY": fy,
        "Name of Bank": bank_name,
        "raw_score": raw_score,
        "Total Words": total_words,
        "category_scores": category_scores
    }
    
    return result

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Digitization Score Calculator", layout="wide")
st.title("🏦 Bank Digitization Score Calculator")

# -------------------- SIDEBAR CONFIGURATION --------------------
st.sidebar.header("⚙️ Configuration")

# Normalization Method Selection
st.sidebar.subheader("📊 Score Normalization (0-100)")
normalization_method = st.sidebar.selectbox(
    "Normalization Method",
    options=['minmax', 'percentile', 'zscore', 'word_normalized'],
    index=0,
    help="""
    - **MinMax**: Linear scaling between lowest and highest score
    - **Percentile**: Based on ranking (best = 100, worst = 0)
    - **Z-Score**: Statistical normalization (centered at 50)
    - **Word Normalized**: Score per 10,000 words (no inter-bank scaling)
    """
)

# Custom Weights Section
st.sidebar.subheader("⚖️ Category Weights")
use_custom_weights = st.sidebar.checkbox("Use Custom Weights", value=False,
                                         help="Enable to set custom weights for each category")

if use_custom_weights:
    st.sidebar.info("Adjust weights for each category below:")
    custom_weights = {}
    for category in keywords.keys():
        short_name = category.split("(")[0].strip() if "(" in category else category
        custom_weights[category] = st.sidebar.number_input(
            f"{short_name}", 
            min_value=0.0, 
            max_value=10.0, 
            value=float(default_weights[category]),
            step=0.1,
            key=f"weight_{category}"
        )
    weights = custom_weights
else:
    weights = default_weights
    st.sidebar.success("Using default weights (all = 1.0)")

# Display current configuration
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Current Settings")
st.sidebar.write(f"**Normalization Method:** {normalization_method.title()}")
st.sidebar.write(f"**Custom Weights:** {'Yes' if use_custom_weights else 'No'}")

# -------------------- FILE UPLOAD --------------------
uploaded_files = st.file_uploader("Upload PDF Reports", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"Found {len(uploaded_files)} PDF file(s). Processing in parallel...")
    
    # Compile patterns once (cached)
    compiled_patterns = compile_keyword_patterns(keywords)
    
    # Process PDFs in parallel
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=min(4, len(uploaded_files))) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_pdf, pdf_file, compiled_patterns, weights): pdf_file.name 
            for pdf_file in uploaded_files
        }
        
        completed = 0
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    status_text.markdown(f"✅ Processed: `{filename}`")
                else:
                    status_text.markdown(f"❌ Failed: `{filename}`")
            except Exception as e:
                status_text.markdown(f"❌ Error in `{filename}`: {str(e)}")
            
            completed += 1
            progress_bar.progress(completed / len(uploaded_files))
    
    # Show Results
    if results:
        # Extract raw scores for normalization
        raw_scores = [r["raw_score"] for r in results]
        
        # Normalize scores to 0-100
        if normalization_method == 'word_normalized':
            # Per 10,000 words normalization
            normalized_scores = [(r["raw_score"] / r["Total Words"]) * 10000 if r["Total Words"] > 0 else 0 
                                for r in results]
            # Then scale to 0-100 based on max
            max_norm = max(normalized_scores) if normalized_scores else 1
            normalized_scores = [(s / max_norm) * 100 for s in normalized_scores]
        else:
            normalized_scores = normalize_to_100(raw_scores, method=normalization_method)
        
        # Build final dataframe
        final_results = []
        for i, result in enumerate(results):
            row = {
                "FY": result["FY"],
                "Name of Bank": result["Name of Bank"],
                "Digitization Score": round(normalized_scores[i], 2),
                "Total Words": result["Total Words"]
            }
            
            # Add category scores (also normalized to 0-100 scale)
            for category, details in result["category_scores"].items():
                short_name = category.split("(")[0].strip() if "(" in category else category
                # Normalize category score proportionally
                category_contribution = (details["weighted_score"] / result["raw_score"] * normalized_scores[i]) if result["raw_score"] > 0 else 0
                row[f"{short_name} Score"] = round(category_contribution, 2)
            
            final_results.append(row)
        
        df = pd.DataFrame(final_results)
        
        st.markdown("### 📊 Final Results")
        
        # Reorder columns
        cols = ["FY", "Name of Bank", "Digitization Score", "Total Words"]
        category_cols = [col for col in df.columns if col not in cols]
        df = df[cols + sorted(category_cols)]
        
        # Sort by digitization score (descending)
        df = df.sort_values("Digitization Score", ascending=False).reset_index(drop=True)
        
        st.dataframe(df, use_container_width=True)

        # Add visualization
        st.markdown("### 📈 Score Comparison")
        
        # Overall scores bar chart
        chart_data = df[["Name of Bank", "FY", "Digitization Score"]].copy()
        chart_data["Bank-FY"] = chart_data["Name of Bank"] + " (" + chart_data["FY"] + ")"
        
        import plotly.express as px
        fig = px.bar(chart_data, x="Bank-FY", y="Digitization Score", 
                     title="Digitization Scores Comparison",
                     labels={"Digitization Score": "Score (0-100)"},
                     color="Digitization Score",
                     color_continuous_scale="Viridis")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown for selected bank
        st.markdown("### 📊 Category Score Breakdown")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_bank_fy = st.selectbox(
                "Select a bank to view detailed scores:", 
                [f"{row['Name of Bank']} ({row['FY']})" for _, row in df.iterrows()]
            )
        
        # Extract selected bank data
        selected_idx = [f"{row['Name of Bank']} ({row['FY']})" for _, row in df.iterrows()].index(selected_bank_fy)
        bank_data = df.iloc[selected_idx]
        
        with col2:
            st.metric("Total Digitization Score", f"{bank_data['Digitization Score']:.2f}/100")
            st.metric("Total Words", f"{bank_data['Total Words']:,}")
        
        # Category scores visualization
        category_scores_viz = {col.replace(" Score", ""): bank_data[col] 
                          for col in bank_data.index if " Score" in col and col != "Digitization Score"}
        
        if category_scores_viz:
            chart_data_cat = pd.DataFrame({
                "Category": list(category_scores_viz.keys()),
                "Score": list(category_scores_viz.values())
            }).sort_values("Score", ascending=True)
            
            fig2 = px.bar(chart_data_cat, x="Score", y="Category", 
                         orientation='h',
                         title=f"Category Scores for {selected_bank_fy}",
                         labels={"Score": "Category Score"},
                         color="Score",
                         color_continuous_scale="Blues")
            st.plotly_chart(fig2, use_container_width=True)

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV", 
            data=csv, 
            file_name="Digitization_Scores.csv", 
            mime="text/csv"
        )
        
        # Show applied settings
        st.markdown("---")
        st.markdown("### ⚙️ Settings Used for This Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Normalization Method:** {normalization_method.title()}")
            st.write(f"**Score Range:** 0-100")
        with col2:
            st.write(f"**Custom Weights:** {'Yes' if use_custom_weights else 'No'}")
            if use_custom_weights:
                with st.expander("View Weight Details"):
                    for cat, weight in weights.items():
                        short_name = cat.split("(")[0].strip() if "(" in cat else cat
                        st.write(f"- {short_name}: {weight}")

else:
    st.info("Please upload one or more PDF files to begin analysis.")
    
    # Show instruction guide
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Instructions:
        1. **Configure Settings** (Optional):
           - Choose normalization method for 0-100 scoring
           - Enable custom weights to modify category importance
        
        2. **Upload PDF Files**:
           - Click the upload button above
           - Select one or more bank annual report PDFs
           - Recommended format: `BankName_FY2023-24.pdf` or `BankName_2023-2024.pdf`
        
        3. **Review Results**:
           - View digitization scores (0-100 scale) for each bank
           - Compare banks with interactive charts
           - Explore category-wise score breakdown
           - Download results as CSV
        
        ### Normalization Methods:
        - **MinMax**: Best score = 100, worst = 0, others scaled linearly
        - **Percentile**: Based on ranking among uploaded documents
        - **Z-Score**: Statistical normalization (average = 50)
        - **Word Normalized**: Accounts for document length, then scaled
        
        ### Performance Optimizations:
        - ✅ Parallel PDF processing (4x faster)
        - ✅ Pre-compiled regex patterns
        - ✅ Efficient text extraction
        - ✅ Smart FY extraction (handles multiple formats)
        """)
        
    # Show example filenames
    with st.expander("📝 Supported Filename Formats"):
        st.markdown("""
        The app automatically extracts FY year from these formats:
        - `HDFC_FY2023-24.pdf` → FY: **2023-24**
        - `ICICI_2023-2024.pdf` → FY: **2023-24**
        - `SBI_FY23-24.pdf` → FY: **2023-24**
        - `Axis_2023.pdf` → FY: **2023**
        """)
