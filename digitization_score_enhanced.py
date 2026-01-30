import os
import pandas as pd
from pypdf import PdfReader
import re
import streamlit as st
import hashlib
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

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

# -------------------- PRE-COMPILED REGEX PATTERNS --------------------
@st.cache_data
def compile_keyword_patterns(keywords_dict):
    """Pre-compile regex patterns for all keywords"""
    patterns = {}
    for category, keyword_list in keywords_dict.items():
        patterns[category] = {}
        for keyword in keyword_list:
            pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b')
            patterns[category][keyword] = pattern
    return patterns

# Compile patterns once
KEYWORD_PATTERNS = compile_keyword_patterns(keywords)

# -------------------- PERFORMANCE OPTIMIZED FUNCTIONS --------------------
@st.cache_data
def extract_text_from_pdf(pdf_bytes, filename):
    """Cached PDF text extraction with error handling"""
    try:
        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text.lower() if text else None
    except Exception as e:
        st.warning(f"⚠️ Error extracting text from {filename}: {str(e)}")
        return None

def fast_word_count(text):
    """Fast word counting without regex"""
    return len(text.split())

def assess_text_quality(text, word_count):
    """Assess PDF text extraction quality"""
    if word_count < 1000:
        return "Poor", "⚠️ Very short document (< 1,000 words)"
    
    # Check for gibberish or scanned PDF indicators
    avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
    
    if avg_word_length > 15:
        return "Poor", "⚠️ Possible scanned/OCR issues"
    elif word_count < 5000:
        return "Medium", "⚡ Short document"
    else:
        return "Good", "✅ Good quality"

def compute_digitization_score_optimized(text, weights, normalize=True, normalization_factor=1000, max_density=0.1):
    """
    Optimized single-pass category scoring with explainability metrics
    
    Args:
        text: Extracted text (lowercase)
        weights: Category weights dict
        normalize: Whether to normalize by word count
        normalization_factor: Multiplier for normalized scores
        max_density: Maximum keyword density cap (default 10%)
    """
    total_words = fast_word_count(text)
    category_details = {}
    total_score = 0
    all_matches = []

    for category, keyword_list in keywords.items():
        category_matches = {}
        category_score = 0
        
        # Single pass through keywords using pre-compiled patterns
        for keyword in keyword_list:
            pattern = KEYWORD_PATTERNS[category][keyword]
            matches = pattern.findall(text)
            count = len(matches)
            if count > 0:
                category_matches[keyword] = count
                category_score += count
        
        # Apply keyword density cap to prevent stuffing bias
        max_allowed_matches = int(total_words * max_density)
        if category_score > max_allowed_matches:
            original_score = category_score
            category_score = max_allowed_matches
            capped = True
        else:
            capped = False
        
        weighted_score = category_score * weights[category]
        
        if normalize and total_words > 0:
            normalized_score = (weighted_score / total_words) * normalization_factor
        else:
            normalized_score = weighted_score
        
        # Get top matched keywords
        top_keywords = sorted(category_matches.items(), key=lambda x: x[1], reverse=True)[:5]
        
        category_details[category] = {
            "raw_count": category_score,
            "weighted_score": weighted_score,
            "normalized_score": round(normalized_score, 2),
            "top_keywords": top_keywords,
            "unique_keywords_matched": len(category_matches),
            "capped": capped
        }
        total_score += normalized_score

    # Calculate category contributions
    for category in category_details:
        if total_score > 0:
            contribution = (category_details[category]["normalized_score"] / total_score) * 100
            category_details[category]["contribution_percent"] = round(contribution, 1)
        else:
            category_details[category]["contribution_percent"] = 0.0

    return round(total_score, 2), category_details, total_words

def process_single_pdf(pdf_file, weights, use_normalization, normalization_factor, max_density):
    """Process a single PDF file"""
    filename = pdf_file.name
    pdf_bytes = pdf_file.read()
    
    text = extract_text_from_pdf(pdf_bytes, filename)
    
    if not text:
        return None
    
    # Extract bank name and FY from filename
    filename_parts = os.path.splitext(filename)[0].split("_")
    bank_name = filename_parts[0]
    
    fy = "Unknown"
    for part in filename_parts:
        if "FY" in part or re.search(r'(20\d{2})[-_](\d{2,4})', part):
            fy = part
            if fy.startswith("FY"):
                fy = fy[2:]
            break
    
    total_score, category_scores, word_count = compute_digitization_score_optimized(
        text, weights, use_normalization, normalization_factor, max_density
    )
    
    # Assess text quality
    quality, quality_note = assess_text_quality(text, word_count)
    
    # Create result dictionary
    result = {
        "FY": fy,
        "Name of Bank": bank_name,
        "Digitization Score": total_score,
        "Total Words": word_count,
        "Text Quality": quality,
        "Quality Note": quality_note,
        "category_details": category_scores  # Store for explainability
    }
    
    # Add individual category scores
    for category, details in category_scores.items():
        short_name = category.split("(")[0].strip() if "(" in category else category
        result[f"{short_name} Score"] = details["normalized_score"]
        result[f"{short_name} Contribution %"] = details["contribution_percent"]
    
    return result

def process_pdfs_parallel(uploaded_files, weights, use_normalization, normalization_factor, max_density):
    """Process multiple PDFs in parallel (for 10+ files)"""
    results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(process_single_pdf, pdf_file, weights, use_normalization, 
                          normalization_factor, max_density): pdf_file.name 
            for pdf_file in uploaded_files
        }
        
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                st.error(f"❌ Error processing {filename}: {str(e)}")
    
    return results

def calculate_percentiles(df, score_column="Digitization Score"):
    """Calculate percentile ranks for all banks"""
    if len(df) == 0:
        return df
    
    df['Percentile Rank'] = df[score_column].rank(pct=True) * 100
    df['Percentile Rank'] = df['Percentile Rank'].round(1)
    return df

def generate_config_hash(keywords_dict, weights, normalization, norm_factor, max_density):
    """Generate reproducible hash for configuration"""
    config = {
        "keywords": sorted([(k, sorted(v)) for k, v in keywords_dict.items()]),
        "weights": sorted(weights.items()),
        "normalization": normalization,
        "norm_factor": norm_factor,
        "max_density": max_density
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Digitization Score Calculator", layout="wide")

# Initialize session state for results persistence
if 'results' not in st.session_state:
    st.session_state.results = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'config_hash' not in st.session_state:
    st.session_state.config_hash = None
if 'timestamp' not in st.session_state:
    st.session_state.timestamp = None

st.title("🏦 Bank Digitization Score Calculator")
st.markdown("*Advanced PDF Analysis with Explainability & Quality Control*")

# -------------------- SIDEBAR CONFIGURATION --------------------
st.sidebar.header("⚙️ Configuration")

# Normalization Toggle
st.sidebar.subheader("📊 Normalization Settings")
use_normalization = st.sidebar.checkbox("Enable Word Count Normalization", value=True, 
                                        help="When enabled, scores are normalized by total word count")

normalization_factor = 1000
if use_normalization:
    normalization_factor = st.sidebar.number_input(
        "Normalization Factor", 
        min_value=1, 
        max_value=10000, 
        value=1000,
        step=100,
        help="Multiplier for normalized scores (default: 1000)"
    )

# Keyword Density Cap
st.sidebar.subheader("🛡️ Quality Control")
max_density = st.sidebar.slider(
    "Max Keyword Density (%)",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
    help="Cap keyword matches to prevent keyword stuffing bias"
) / 100

min_word_threshold = st.sidebar.number_input(
    "Minimum Word Threshold",
    min_value=0,
    max_value=10000,
    value=1000,
    step=100,
    help="Warn if document has fewer words than this"
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

# Generate configuration hash
config_hash = generate_config_hash(keywords, weights, use_normalization, normalization_factor, max_density)

# Display current configuration
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Current Configuration")
st.sidebar.write(f"**Config Hash:** `{config_hash}`")
st.sidebar.write(f"**Normalization:** {'Enabled' if use_normalization else 'Disabled'}")
if use_normalization:
    st.sidebar.write(f"**Factor:** {normalization_factor}")
st.sidebar.write(f"**Max Density:** {max_density*100}%")
st.sidebar.write(f"**Custom Weights:** {'Yes' if use_custom_weights else 'No'}")

# -------------------- FILE UPLOAD --------------------
uploaded_files = st.file_uploader("Upload PDF Reports", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    st.info(f"📁 Found {num_files} PDF file(s). Processing...")
    
    # Use parallel processing for 10+ files
    use_parallel = num_files >= 10
    if use_parallel:
        st.info("⚡ Using parallel processing for faster analysis...")
    
    progress_bar = st.progress(0)
    
    with st.spinner("Processing PDFs..."):
        if use_parallel:
            results = process_pdfs_parallel(uploaded_files, weights, use_normalization, 
                                          normalization_factor, max_density)
            progress_bar.progress(1.0)
        else:
            results = []
            for i, pdf_file in enumerate(uploaded_files):
                result = process_single_pdf(pdf_file, weights, use_normalization, 
                                          normalization_factor, max_density)
                if result:
                    results.append(result)
                    bank_name = result["Name of Bank"]
                    fy = result["FY"]
                    score = result["Digitization Score"]
                    quality = result["Text Quality"]
                    st.success(f"✅ `{bank_name}` ({fy}) → Score: **{score}** | Quality: **{quality}**")
                
                progress_bar.progress((i + 1) / num_files)
    
    # Store results in session state
    if results:
        st.session_state.results = results
        st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.config_hash = config_hash
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Extract category details for later use
        category_details_map = {row["Name of Bank"]: row["category_details"] for row in results}
        
        # Remove category_details from display DataFrame
        display_df = df.drop(columns=["category_details", "Quality Note"], errors='ignore')
        
        # Calculate percentiles
        display_df = calculate_percentiles(display_df)
        
        # Reorder columns
        base_cols = ["FY", "Name of Bank", "Digitization Score", "Percentile Rank", 
                    "Total Words", "Text Quality"]
        category_score_cols = sorted([col for col in display_df.columns if " Score" in col and col != "Digitization Score"])
        contribution_cols = sorted([col for col in display_df.columns if "Contribution %" in col])
        other_cols = [col for col in display_df.columns if col not in base_cols + category_score_cols + contribution_cols]
        
        display_df = display_df[base_cols + category_score_cols + contribution_cols + other_cols]
        
        st.session_state.results_df = display_df
        st.session_state.category_details_map = category_details_map

# Display results if available in session state
if st.session_state.results_df is not None:
    df = st.session_state.results_df
    category_details_map = st.session_state.category_details_map
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    
    # Display timestamp and config
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Analysis Date", st.session_state.timestamp)
    with col2:
        st.metric("Configuration Hash", st.session_state.config_hash)
    with col3:
        st.metric("Banks Analyzed", len(df))
    
    # Main results table
    st.markdown("#### 📈 Overall Scores & Rankings")
    st.dataframe(df, use_container_width=True, height=400)
    
    # Quality warnings
    poor_quality = df[df["Text Quality"] == "Poor"]
    if len(poor_quality) > 0:
        st.warning(f"⚠️ {len(poor_quality)} document(s) with poor text quality detected. Results may be unreliable.")
        with st.expander("View Poor Quality Documents"):
            st.dataframe(poor_quality[["Name of Bank", "FY", "Total Words", "Text Quality"]])
    
    # -------------------- EXPLAINABILITY SECTION --------------------
    st.markdown("---")
    st.markdown("### 🔍 Detailed Explainability")
    
    selected_bank = st.selectbox("Select a bank for detailed analysis:", df["Name of Bank"].unique())
    
    if selected_bank:
        bank_data = df[df["Name of Bank"] == selected_bank].iloc[0]
        category_details = category_details_map[selected_bank]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Score", f"{bank_data['Digitization Score']:.2f}")
        with col2:
            st.metric("Percentile Rank", f"{bank_data['Percentile Rank']:.1f}%")
        with col3:
            st.metric("Total Words", f"{bank_data['Total Words']:,}")
        with col4:
            st.metric("Text Quality", bank_data["Text Quality"])
        
        # Category breakdown tabs
        tab1, tab2, tab3 = st.tabs(["📊 Category Scores", "🔑 Top Keywords", "📈 Contribution Analysis"])
        
        with tab1:
            # Category scores bar chart
            category_scores_viz = {col.replace(" Score", ""): bank_data[col] 
                                  for col in bank_data.index if " Score" in col and col != "Digitization Score"}
            
            chart_data = pd.DataFrame({
                "Category": list(category_scores_viz.keys()),
                "Score": list(category_scores_viz.values())
            }).sort_values("Score", ascending=False)
            
            st.bar_chart(chart_data.set_index("Category"), use_container_width=True)
            
            # Detailed category table
            st.markdown("##### Category Details")
            category_df = pd.DataFrame([
                {
                    "Category": cat.split("(")[0].strip() if "(" in cat else cat,
                    "Raw Count": details["raw_count"],
                    "Weighted Score": f"{details['weighted_score']:.2f}",
                    "Normalized Score": f"{details['normalized_score']:.2f}",
                    "Unique Keywords": details["unique_keywords_matched"],
                    "Capped": "Yes" if details["capped"] else "No"
                }
                for cat, details in category_details.items()
            ])
            st.dataframe(category_df, use_container_width=True)
        
        with tab2:
            # Top keywords for each category
            st.markdown("##### Top Matched Keywords by Category")
            for category, details in category_details.items():
                short_name = category.split("(")[0].strip() if "(" in category else category
                top_kw = details["top_keywords"]
                
                if top_kw:
                    with st.expander(f"**{short_name}** ({details['raw_count']} matches)"):
                        for keyword, count in top_kw:
                            st.write(f"- **{keyword}**: {count} occurrence(s)")
                else:
                    with st.expander(f"**{short_name}** (0 matches)"):
                        st.write("No keywords matched")
        
        with tab3:
            # Contribution pie chart
            contribution_data = {
                cat.split("(")[0].strip() if "(" in cat else cat: details["contribution_percent"]
                for cat, details in category_details.items()
                if details["contribution_percent"] > 0
            }
            
            if contribution_data:
                contrib_df = pd.DataFrame({
                    "Category": list(contribution_data.keys()),
                    "Contribution %": list(contribution_data.values())
                }).sort_values("Contribution %", ascending=False)
                
                st.dataframe(contrib_df, use_container_width=True)
                
                # Bar chart of contributions
                st.bar_chart(contrib_df.set_index("Category"), use_container_width=True)
    
    # -------------------- COMPARISON SECTION --------------------
    st.markdown("---")
    st.markdown("### 🔄 Side-by-Side Comparison")
    
    if len(df) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            bank1 = st.selectbox("Select first bank:", df["Name of Bank"].unique(), key="bank1")
        with col2:
            bank2_options = [b for b in df["Name of Bank"].unique() if b != bank1]
            bank2 = st.selectbox("Select second bank:", bank2_options, key="bank2")
        
        if bank1 and bank2:
            bank1_data = df[df["Name of Bank"] == bank1].iloc[0]
            bank2_data = df[df["Name of Bank"] == bank2].iloc[0]
            
            # Comparison metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"{bank1} Score", 
                    f"{bank1_data['Digitization Score']:.2f}",
                    delta=f"{bank1_data['Digitization Score'] - bank2_data['Digitization Score']:.2f}"
                )
            with col2:
                st.metric(
                    f"{bank2} Score",
                    f"{bank2_data['Digitization Score']:.2f}"
                )
            with col3:
                score_diff_pct = ((bank1_data['Digitization Score'] - bank2_data['Digitization Score']) / 
                                 bank2_data['Digitization Score'] * 100)
                st.metric("Difference", f"{abs(score_diff_pct):.1f}%")
            
            # Category comparison
            st.markdown("##### Category-wise Comparison")
            category_cols = [col for col in df.columns if " Score" in col and col != "Digitization Score"]
            
            comparison_data = []
            for col in category_cols:
                cat_name = col.replace(" Score", "")
                comparison_data.append({
                    "Category": cat_name,
                    bank1: bank1_data[col],
                    bank2: bank2_data[col],
                    "Difference": bank1_data[col] - bank2_data[col]
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("Upload at least 2 PDFs to enable comparison.")
    
    # -------------------- EXPORT SECTION --------------------
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export with metadata
        metadata_rows = [
            ["Analysis Metadata", ""],
            ["Timestamp", st.session_state.timestamp],
            ["Configuration Hash", st.session_state.config_hash],
            ["Normalization Enabled", str(use_normalization)],
            ["Normalization Factor", str(normalization_factor)],
            ["Max Keyword Density", f"{max_density*100}%"],
            ["Custom Weights", str(use_custom_weights)],
            [""],
            ["Category Weights", "Value"]
        ]
        
        for cat, weight in weights.items():
            short_name = cat.split("(")[0].strip() if "(" in cat else cat
            metadata_rows.append([short_name, weight])
        
        metadata_rows.append([""])
        metadata_rows.append(["Results", ""])
        
        # Combine metadata and results
        metadata_df = pd.DataFrame(metadata_rows)
        
        # Create export CSV
        from io import StringIO
        csv_buffer = StringIO()
        metadata_df.to_csv(csv_buffer, index=False, header=False)
        csv_buffer.write("\n")
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        st.download_button(
            "⬇️ Download Results as CSV (with metadata)", 
            data=csv_content, 
            file_name=f"Digitization_Scores_{config_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            mime="text/csv"
        )
    
    with col2:
        # JSON export for reproducibility
        export_data = {
            "metadata": {
                "timestamp": st.session_state.timestamp,
                "config_hash": st.session_state.config_hash,
                "normalization_enabled": use_normalization,
                "normalization_factor": normalization_factor,
                "max_keyword_density": max_density,
                "custom_weights": use_custom_weights,
                "weights": weights
            },
            "results": st.session_state.results
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            "⬇️ Download Complete Analysis (JSON)",
            data=json_str,
            file_name=f"Analysis_{config_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

else:
    if not uploaded_files:
        st.info("Please upload one or more PDF files to begin analysis.")
        
        # Show instruction guide
        with st.expander("📖 How to Use This Tool"):
            st.markdown("""
            ### Instructions:
            
            1. **Configure Settings** (Optional):
               - Use the sidebar to enable/disable word count normalization
               - Adjust the normalization factor if needed
               - Set keyword density cap to prevent keyword stuffing
               - Enable custom weights to modify category importance
            
            2. **Upload PDF Files**:
               - Click the upload button above
               - Select one or more bank annual report PDFs
               - File names should follow format: `BankName_FY20XX.pdf`
               - For 10+ files, parallel processing is automatically enabled
            
            3. **Review Results**:
               - View digitization scores and percentile ranks
               - Explore category-wise breakdowns
               - Analyze top matched keywords
               - Compare banks side-by-side
               - Check text quality indicators
            
            4. **Export Data**:
               - Download results as CSV with full metadata
               - Export complete analysis as JSON for reproducibility
            
            ### Features:
            
            **Performance**:
            - ⚡ Cached PDF text extraction
            - 🚀 Pre-compiled regex patterns
            - 🔄 Parallel processing for large batches
            - 💾 Session memory (results persist across reruns)
            
            **Quality Control**:
            - 📊 Text quality assessment (Good/Medium/Poor)
            - ⚠️ Minimum word threshold warnings
            - 🛡️ Keyword density capping
            - 🔍 Scanned PDF detection
            
            **Explainability**:
            - 📈 Per-category raw counts, weighted scores, and normalized scores
            - 🔑 Top 5 matched keywords per category
            - 📊 Category contribution percentages
            - 💡 Full transparency in scoring
            
            **Comparability**:
            - 📊 Percentile rankings across all banks
            - 🔄 Side-by-side bank comparison
            - 📅 Automatic FY extraction from filenames
            
            **Reproducibility**:
            - 🔐 Configuration hash ID
            - ⏰ Timestamp of analysis
            - 📥 Export settings with results
            - 🔄 Re-run with exact same parameters
            
            ### About Normalization:
            - **Enabled**: Scores are normalized per 1,000 words (accounts for document length)
            - **Disabled**: Raw weighted keyword counts are used
            - **Factor**: Adjustable multiplier (default: 1000)
            
            ### About Keyword Density Cap:
            - Prevents bias from keyword stuffing
            - Default: 10% (max 10% of total words can be matched keywords)
            - Flagged documents show "Capped: Yes" in category details
            
            ### About Custom Weights:
            - Default weight for all categories is 1.0
            - Increase weight to prioritize certain categories
            - Set weight to 0 to exclude a category from scoring
            - All weight changes are tracked in the config hash
            """)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>🏦 Bank Digitization Score Calculator | Version 2.0 | Enhanced with Performance & Explainability</p>
        <p>📊 Features: Caching • Parallel Processing • Quality Control • Explainability • Reproducibility</p>
    </div>
    """,
    unsafe_allow_html=True
)
