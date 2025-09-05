# streamlit_app.py
import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import time
import os

# Try to import modules with error handling for Streamlit Cloud
try:
    from langchain_foundation import PharmaKnowledgeBase, PharmaRAGChains
    from langgraph_agents import create_pharma_workflow, PharmaLaunchState
    from langsmith_monitoring import PharmaLangSmithMonitor, create_monitored_pharma_workflow
    MODULES_AVAILABLE = True
except Exception as e:
    st.error(f"Module import error: {e}")
    MODULES_AVAILABLE = False

# For Streamlit Cloud - use st.secrets instead of config.py
class Config:
    """Configuration class that works with Streamlit secrets"""
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    LLM_MODEL = "gpt-4-turbo"
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    LANGCHAIN_TRACING_V2 = st.secrets.get("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "true"))
    LANGCHAIN_ENDPOINT = st.secrets.get("LANGCHAIN_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))
    LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY"))
    LANGCHAIN_PROJECT = st.secrets.get("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "pharma-launch-assistant"))
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
    VECTOR_STORE_PATH = "./data/vector_store"
    
    FDA_BASE_URL = "https://www.fda.gov"
    CLINICALTRIALS_API = "https://clinicaltrials.gov/api"

# Set environment variables for LangChain
if Config.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
if Config.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = Config.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = Config.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"] = Config.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_PROJECT"] = Config.LANGCHAIN_PROJECT

# Page configuration
st.set_page_config(
    page_title="Pharma Launch Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the pharma launch system with caching"""
    try:
        if MODULES_AVAILABLE:
            monitor = PharmaLangSmithMonitor()
            return monitor, None, None
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error initializing monitoring: {e}")
        return None, None, None

@st.cache_data
def load_sample_data():
    """Load sample pharmaceutical data for demonstration"""
    return {
        "drug_classes": [
            "ACE Inhibitors", "Beta Blockers", "Statins", "PPI", "SSRI",
            "Immunotherapy", "Monoclonal Antibodies", "Tyrosine Kinase Inhibitors",
            "GLP-1 Agonists", "SGLT-2 Inhibitors"
        ],
        "indications": [
            "Hypertension", "Diabetes", "Cancer", "Depression", "Anxiety",
            "Cardiovascular Disease", "Inflammatory Disorders", "Neurological Disorders",
            "Infectious Diseases", "Metabolic Disorders"
        ],
        "sample_products": [
            {"name": "CardioMax", "class": "ACE Inhibitors", "indication": "Hypertension"},
            {"name": "DiabetesCure", "class": "GLP-1 Agonists", "indication": "Diabetes"}, 
            {"name": "OncoTarget", "class": "Monoclonal Antibodies", "indication": "Cancer"},
            {"name": "MoodLift", "class": "SSRI", "indication": "Depression"}
        ]
    }

def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">💊 Pharmaceutical Launch Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Check if API keys are configured
    if not Config.OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not configured. Please add it to Streamlit secrets.")
        st.info("Add your API keys in the Streamlit Cloud dashboard under 'Secrets management'.")
        return False
    
    # Description
    st.markdown("""
    **Powered by LangChain, LangGraph & LangSmith**
    
    This advanced AI system helps pharmaceutical companies analyze market opportunities, 
    assess competitive landscapes, navigate regulatory requirements, and develop comprehensive 
    launch strategies for new products.
    """)
    
    if not MODULES_AVAILABLE:
        st.warning("⚠️ Some modules are not available. Running in demo mode.")
    
    return True

def render_sidebar():
    """Render the sidebar with system controls"""
    st.sidebar.title("🔧 System Configuration")
    
    # System status
    st.sidebar.markdown("### System Status")
    
    if Config.OPENAI_API_KEY:
        st.sidebar.success("✅ OpenAI API: Configured")
    else:
        st.sidebar.error("❌ OpenAI API: Not configured")
    
    if Config.LANGCHAIN_API_KEY and MODULES_AVAILABLE:
        st.sidebar.success("✅ LangSmith: Configured")
        monitor, kb, workflow = initialize_system()
    else:
        st.sidebar.warning("⚠️ LangSmith: Not configured or modules unavailable")
        monitor = None
    
    # Sample data toggle
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True, help="Use pre-loaded sample data for demonstration")
    
    # Model settings
    st.sidebar.markdown("### Model Configuration")
    model_temp = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.1, 0.1)
    
    # Data source settings  
    st.sidebar.markdown("### Data Sources")
    use_fda_data = st.sidebar.checkbox("FDA Guidance Documents", value=True)
    use_clinical_trials = st.sidebar.checkbox("ClinicalTrials.gov", value=True)
    use_market_data = st.sidebar.checkbox("Market Intelligence", value=False, disabled=True, help="Coming soon")
    
    return {
        "use_sample_data": use_sample_data,
        "model_temperature": model_temp,
        "use_fda_data": use_fda_data,
        "use_clinical_trials": use_clinical_trials,
        "monitor": monitor
    }

def render_product_input():
    """Render product input form"""
    st.markdown("## 📝 Product Information")
    
    sample_data = load_sample_data()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        product_name = st.text_input(
            "Product Name", 
            value="ImmunoAdvance",
            help="Enter the name of your pharmaceutical product"
        )
        
    with col2:
        drug_class = st.selectbox(
            "Drug Class",
            options=sample_data["drug_classes"],
            index=5,  # Default to Immunotherapy
            help="Select the therapeutic class of your drug"
        )
    
    with col3:
        indication = st.selectbox(
            "Primary Indication", 
            options=sample_data["indications"],
            index=2,  # Default to Cancer
            help="Select the primary medical condition being treated"
        )
    
    # Additional details
    st.markdown("### Additional Details")
    col4, col5 = st.columns(2)
    
    with col4:
        development_stage = st.selectbox(
            "Development Stage",
            ["Preclinical", "Phase I", "Phase II", "Phase III", "Filing/Review", "Pre-Launch"],
            index=4
        )
        
    with col5:
        target_market = st.multiselect(
            "Target Markets",
            ["US", "EU", "Japan", "China", "Canada", "Australia", "Brazil"],
            default=["US", "EU"]
        )
    
    launch_timeline = st.slider(
        "Expected Launch Timeline (months)",
        6, 60, 18,
        help="Expected time to launch from current stage"
    )
    
    return {
        "product_name": product_name,
        "drug_class": drug_class, 
        "indication": indication,
        "development_stage": development_stage,
        "target_market": target_market,
        "launch_timeline": launch_timeline
    }

def create_mock_analysis_result(product_info):
    """Create a comprehensive mock analysis result for demo purposes"""
    return {
        "product_name": product_info["product_name"],
        "final_report": f"""
# Pharmaceutical Launch Analysis Report

## EXECUTIVE SUMMARY

**Product:** {product_info['product_name']} ({product_info['drug_class']})  
**Indication:** {product_info['indication']}  
**Development Stage:** {product_info['development_stage']}

### Key Findings:
- **Strong market opportunity** identified in {product_info['indication']} treatment
- **Competitive landscape** shows room for differentiation with premium positioning
- **Regulatory pathway** is well-established for {product_info['drug_class']} class
- **Clinical development** strategy should focus on superiority trials
- **Market access** considerations require early payer engagement

### Strategic Recommendations:
1. **Accelerate Phase III development** focusing on primary efficacy endpoint
2. **Develop premium pricing strategy** based on clinical differentiation
3. **Establish early market access** discussions with key payers
4. **Build KOL relationships** in target therapeutic area
5. **Plan specialty care** distribution strategy

### Investment Requirements: $150-200M
### Timeline to Launch: 18-24 months  
### Peak Sales Potential: $500M-1B annually

## DETAILED ANALYSIS

### Market Opportunity Assessment
The {product_info['indication']} market represents a significant commercial opportunity with:
- Growing patient population and unmet medical needs
- Limited therapeutic options in current standard of care
- Strong payer willingness to reimburse effective treatments
- Opportunity for premium pricing with demonstrated clinical benefit

### Competitive Landscape
Current market leaders include established therapies with known limitations:
- Safety concerns with existing treatments create differentiation opportunity
- Efficacy gaps in certain patient subpopulations
- Limited convenience factors (dosing, administration) present positioning advantages

### Regulatory Strategy
FDA approval pathway for {product_info['drug_class']} is well-defined:
- Standard 505(b)(1) NDA submission recommended
- No anticipated regulatory obstacles based on class precedent
- Post-marketing commitments likely but manageable
- International regulatory alignment feasible

### Clinical Development Strategy  
Phase III program design recommendations:
- Superiority trial design vs. active comparator
- Primary endpoint aligned with regulatory guidance
- Key secondary endpoints supporting commercial positioning
- Patient population enrichment strategy for optimal outcomes

## RISK ASSESSMENT

### Key Risks and Mitigation Strategies:
1. **Clinical Risk:** Trial failure - Mitigate with robust phase II data and adaptive design
2. **Competitive Risk:** New entrants - Accelerate timeline and establish IP protection
3. **Regulatory Risk:** Approval delays - Engage early with FDA and maintain quality standards
4. **Commercial Risk:** Market access - Begin payer discussions during development

### Success Probability: 75-85% based on clinical and regulatory precedent

## RECOMMENDATIONS & NEXT STEPS

### Immediate Actions (0-3 months):
1. Finalize Phase III protocol design
2. Initiate regulatory interactions (Type B meeting)
3. Begin market access strategy development
4. Establish clinical site network

### Medium-term Actions (3-12 months):
1. Commence Phase III enrollment
2. Execute market research and KOL mapping
3. Develop commercial strategy and pricing model
4. Initiate manufacturing scale-up planning

### Long-term Actions (12+ months):
1. Prepare NDA submission package
2. Execute pre-launch commercial activities
3. Establish distribution partnerships
4. Implement launch readiness program

### Decision Points:
- **Month 6:** Phase III interim analysis
- **Month 12:** Commercial strategy finalization  
- **Month 18:** NDA submission decision
- **Month 24:** Launch execution
        """,
        "research_findings": {
            "market_analysis": {
                "query": f"What is the market opportunity for {product_info['indication']}?",
                "insights": f"The {product_info['indication']} market shows strong growth potential with significant unmet medical needs. Current treatments have limitations in efficacy and safety, creating opportunities for innovative {product_info['drug_class']} therapies.",
                "sources": ["Market research reports", "Clinical publications", "Regulatory guidance"]
            },
            "treatment_landscape": {
                "query": f"What are current treatment options for {product_info['indication']}?",
                "insights": f"Standard of care for {product_info['indication']} includes multiple therapeutic classes, but significant gaps remain in patient outcomes and quality of life.",
                "sources": ["Clinical guidelines", "Treatment algorithms", "Real-world evidence"]
            }
        },
        "competitive_analysis": {
            "strategic_recommendations": f"Position {product_info['product_name']} as a best-in-class {product_info['drug_class']} with superior efficacy and safety profile. Focus on premium positioning with specialty care distribution.",
            "detailed_analysis": {
                "competitor_mapping": "Analysis shows 3-4 major competitors with established market share",
                "pricing_strategy": "Premium pricing justified by clinical differentiation",
                "market_positioning": "Target specialty care physicians with superior outcomes messaging"
            }
        },
        "regulatory_assessment": {
            "regulatory_strategy": f"FDA approval pathway for {product_info['drug_class']} is well-established. Recommend 505(b)(1) NDA submission with standard clinical development program.",
            "approval_pathway": "Standard NDA - 505(b)(1)",
            "timeline_estimate": "12-18 months post-submission"
        },
        "clinical_insights": {
            "development_strategy": f"Phase III program should demonstrate superiority over current standard of care in {product_info['indication']}. Recommend enriched patient population and adaptive trial design.",
            "timeline_estimates": "Phase III: 24-36 months",
            "success_factors": "Strong Phase II data, appropriate patient selection, robust statistical plan"
        },
        "market_strategy": {
            "comprehensive_strategy": f"Launch {product_info['product_name']} with premium pricing strategy, focusing on specialty care market initially. Expand to broader markets post-launch based on real-world evidence generation."
        }
    }

def run_analysis(product_info, config):
    """Run the pharmaceutical launch analysis"""
    
    if not Config.OPENAI_API_KEY:
        st.error("OpenAI API key is required to run analysis")
        return None
    
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check if full modules are available
        if not MODULES_AVAILABLE:
            status_text.text("Running in demo mode - using mock analysis...")
            progress_bar.progress(50)
            time.sleep(2)
            
            result = create_mock_analysis_result(product_info)
            progress_bar.progress(100)
            status_text.text("Demo analysis complete!")
            return result
        
        # Build knowledge base
        status_text.text("Building knowledge base...")
        progress_bar.progress(10)
        
        kb = PharmaKnowledgeBase()
        kb.build_knowledge_base(product_info["drug_class"], product_info["indication"])
        progress_bar.progress(30)
        
        # Create workflow
        status_text.text("Initializing multi-agent workflow...")
        if config["monitor"]:
            workflow = create_monitored_pharma_workflow(kb, config["monitor"])
        else:
            workflow = create_pharma_workflow(kb)
        progress_bar.progress(50)
        
        # Prepare initial state
        initial_state = PharmaLaunchState(
            product_name=product_info["product_name"],
            indication=product_info["indication"],
            drug_class=product_info["drug_class"],
            current_agent="research_agent",
            research_findings={},
            competitive_analysis={},
            regulatory_assessment={},
            clinical_insights={},
            market_strategy={},
            final_report="",
            iteration_count=0,
            messages=[]
        )
        
        # Run analysis
        status_text.text("Running multi-agent analysis...")
        progress_bar.progress(70)
        
        result = workflow.invoke(initial_state)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return result
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        
        # Fallback to mock analysis
        st.info("Falling back to demo analysis...")
        return create_mock_analysis_result(product_info)

def render_results(result):
    """Render analysis results with interactive visualizations"""
    if not result:
        return
    
    st.markdown("## 📊 Analysis Results")
    
    # Executive Summary
    st.markdown("### Executive Summary")
    if result.get("final_report"):
        # Extract executive summary from the report
        report_lines = result["final_report"].split('\n')
        exec_summary = []
        in_summary = False
        
        for line in report_lines:
            if "EXECUTIVE SUMMARY" in line.upper():
                in_summary = True
                continue
            elif in_summary and ("DETAILED ANALYSIS" in line.upper() or "## " in line):
                break
            elif in_summary and line.strip():
                exec_summary.append(line)
        
        if exec_summary:
            st.markdown('\n'.join(exec_summary))
        else:
            st.markdown(result["final_report"][:1000] + "..." if len(result["final_report"]) > 1000 else result["final_report"])
    
    # Tabs for detailed results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Research Findings", 
        "🏢 Competitive Analysis", 
        "📋 Regulatory Assessment",
        "🧪 Clinical Strategy", 
        "📈 Market Strategy"
    ])
    
    with tab1:
        st.markdown("### Research Findings")
        if result.get("research_findings"):
            for key, finding in result["research_findings"].items():
                with st.expander(f"Research Point: {finding.get('query', key)}"):
                    st.write(finding.get("insights", "No insights available"))
                    if finding.get("sources"):
                        st.markdown("**Sources:**")
                        for source in finding["sources"]:
                            st.markdown(f"- {source}")
        else:
            st.info("Research findings not available in this analysis run.")
    
    with tab2:
        st.markdown("### Competitive Analysis") 
        if result.get("competitive_analysis"):
            analysis = result["competitive_analysis"]
            
            if isinstance(analysis.get("strategic_recommendations"), str):
                st.markdown("**Strategic Recommendations:**")
                st.write(analysis["strategic_recommendations"])
            
            if analysis.get("detailed_analysis"):
                with st.expander("Detailed Competitive Analysis"):
                    for query, response in analysis["detailed_analysis"].items():
                        st.markdown(f"**{query}**")
                        st.write(response)
        else:
            st.info("Competitive analysis not available in this analysis run.")
    
    with tab3:
        st.markdown("### Regulatory Assessment")
        if result.get("regulatory_assessment"):
            assessment = result["regulatory_assessment"]
            
            if isinstance(assessment.get("regulatory_strategy"), str):
                st.markdown("**Regulatory Strategy:**")
                st.write(assessment["regulatory_strategy"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Approval Pathway", assessment.get("approval_pathway", "TBD"))
            with col2:
                st.metric("Timeline Estimate", assessment.get("timeline_estimate", "TBD"))
        else:
            st.info("Regulatory assessment not available in this analysis run.")
    
    with tab4:
        st.markdown("### Clinical Strategy")
        if result.get("clinical_insights"):
            insights = result["clinical_insights"]
            
            if isinstance(insights.get("development_strategy"), str):
                st.markdown("**Development Strategy:**")
                st.write(insights["development_strategy"])
            
            # Clinical timeline visualization
            st.markdown("**Development Timeline:**")
            phases = ["Phase I", "Phase II", "Phase III", "Filing", "Approval"]
            durations = [12, 24, 36, 12, 12]  # Sample durations in months
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=phases,
                y=durations,
                marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            ))
            fig.update_layout(
                title="Clinical Development Timeline",
                xaxis_title="Development Phase",
                yaxis_title="Duration (Months)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Clinical insights not available in this analysis run.")
    
    with tab5:
        st.markdown("### Market Strategy")
        if result.get("market_strategy"):
            strategy = result["market_strategy"]
            
            if isinstance(strategy.get("comprehensive_strategy"), str):
                st.markdown("**Comprehensive Strategy:**")
                st.write(strategy["comprehensive_strategy"])
            
            # Market opportunity visualization
            st.markdown("**Market Opportunity Analysis:**")
            
            # Sample market data for visualization
            market_data = pd.DataFrame({
                'Market Segment': ['Primary Care', 'Specialty Care', 'Hospital', 'Online'],
                'Market Size ($M)': [500, 300, 200, 50],
                'Growth Rate (%)': [5, 8, 3, 15]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(market_data, values='Market Size ($M)', names='Market Segment', 
                               title='Market Size Distribution')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_bar = px.bar(market_data, x='Market Segment', y='Growth Rate (%)',
                               title='Market Growth Rates by Segment')
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Market strategy not available in this analysis run.")

def render_export_options(result):
    """Render export and sharing options"""
    if not result:
        return
        
    st.markdown("## 📤 Export & Share")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export PDF Report", use_container_width=True):
            st.info("PDF export functionality would be implemented here")
    
    with col2:
        if st.button("📊 Export Data (JSON)", use_container_width=True):
            # Create downloadable JSON
            json_data = json.dumps(result, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"pharma_analysis_{result.get('product_name', 'report')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📧 Email Report", use_container_width=True):
            st.info("Email integration would be implemented here")

def main():
    """Main application function"""
    
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    
    # Render header and check configuration
    if not render_header():
        st.stop()
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    # Main content area
    if not st.session_state.analysis_running:
        # Product input form
        product_info = render_product_input()
        
        # Store product info in session state
        st.session_state.product_info = product_info
        
        # Launch analysis button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Launch Comprehensive Analysis", 
                        use_container_width=True, 
                        type="primary",
                        disabled=not all([product_info["product_name"], 
                                        product_info["drug_class"], 
                                        product_info["indication"]])):
                
                st.session_state.analysis_running = True
                st.rerun()
    
    else:
        # Show analysis in progress
        st.markdown("## 🔄 Analysis in Progress")
        
        # Get product info from session state
        product_info = st.session_state.get("product_info", {
            "product_name": "Test Product",
            "drug_class": "Test Class",
            "indication": "Test Indication"
        })
        
        with st.spinner("Running comprehensive pharmaceutical launch analysis..."):
            # Show progress
            progress_container = st.container()
            
            with progress_container:
                agents_progress = [
                    "🔬 Research Agent: Gathering data...",
                    "🏢 Competitive Analyst: Analyzing competitors...", 
                    "📋 Regulatory Specialist: Assessing requirements...",
                    "🧪 Clinical Strategist: Developing strategy...",
                    "📈 Market Strategist: Creating launch plan...",
                    "📝 Report Writer: Compiling report..."
                ]
                
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                for i, status in enumerate(agents_progress):
                    status_placeholder.info(status)
                    progress_bar.progress((i + 1) / len(agents_progress))
                    time.sleep(1)  # Simulate processing time
                
                # Run actual analysis
                result = run_analysis(product_info, config)
                
                st.session_state.analysis_result = result
                st.session_state.analysis_running = False
                
                status_placeholder.success("✅ Analysis Complete!")
                time.sleep(1)
                st.rerun()
    
    # Display results if available
    if st.session_state.analysis_result:
        render_results(st.session_state.analysis_result)
        
        # Export options
        render_export_options(st.session_state.analysis_result)
        
        # Reset button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                st.session_state.analysis_result = None
                st.session_state.analysis_running = False
                st.rerun()

def render_tutorial_page():
    """Render tutorial page explaining the system"""
    st.markdown("# 📚 Tutorial: Learning LangChain Ecosystem")
    
    st.markdown("""
    ## System Architecture
    
    This pharmaceutical launch intelligence system demonstrates the three core components of the LangChain ecosystem:
    
    ### 🔗 LangChain - Foundation Layer
    - **Document Processing**: Ingests FDA guidance documents, clinical trial data
    - **Vector Storage**: Creates searchable knowledge base using ChromaDB
    - **RAG Chains**: Retrieval-Augmented Generation for domain-specific queries
    - **Specialized Chains**: Regulatory, competitive, and clinical analysis chains
    
    ### 🕸️ LangGraph - Orchestration Layer  
    - **Multi-Agent Workflow**: Coordinates 6 specialized AI agents
    - **State Management**: Maintains analysis state across agent interactions
    - **Sequential Processing**: Research → Competition → Regulatory → Clinical → Market → Report
    - **Error Handling**: Robust error recovery and fallback mechanisms
    
    ### 📊 LangSmith - Observability Layer
    - **Tracing**: Tracks every LLM call and agent interaction
    - **Monitoring**: Real-time performance metrics and cost tracking
    - **Evaluation**: Custom evaluators for pharmaceutical domain accuracy
    - **Debugging**: Detailed logs for optimization and troubleshooting
    
    ## Setup Instructions
    
    ### 1. Configure API Keys
    Add your API keys to Streamlit secrets:
    ```toml
    OPENAI_API_KEY = "your_openai_key"
    LANGCHAIN_API_KEY = "your_langsmith_key"
    LANGCHAIN_TRACING_V2 = "true"
    LANGCHAIN_PROJECT = "pharma-launch-assistant"
    ```
    
    ### 2. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```
    
    ### 3. Run the Application
    ```bash
    streamlit run streamlit_app.py
    ```
    
    ## Key Features
    
    - **Multi-Agent Analysis**: 6 specialized AI agents working together
    - **Domain Expertise**: Deep pharmaceutical industry knowledge
    - **Interactive Visualizations**: Charts and graphs for key insights
    - **Export Capabilities**: Download results in multiple formats
    - **Real-time Monitoring**: Track performance and costs
    """)

# Navigation
def main_navigation():
    """Handle page navigation"""
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["🚀 Launch Analysis", "📚 Tutorial"])
    
    with tab1:
        main()
    
    with tab2:
        render_tutorial_page()

if __name__ == "__main__":
    main_navigation()