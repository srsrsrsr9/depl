# streamlit_app.py
import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import time

from langchain_foundation import PharmaKnowledgeBase, PharmaRAGChains
from langgraph_agents import create_pharma_workflow, PharmaLaunchState
from langsmith_monitoring import PharmaLangSmithMonitor, create_monitored_pharma_workflow

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
        monitor = PharmaLangSmithMonitor()
        return monitor, None, None
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
    
    # Description
    st.markdown("""
    **Powered by LangChain, LangGraph & LangSmith**
    
    This advanced AI system helps pharmaceutical companies analyze market opportunities, 
    assess competitive landscapes, navigate regulatory requirements, and develop comprehensive 
    launch strategies for new products.
    """)

def render_sidebar():
    """Render the sidebar with system controls"""
    st.sidebar.title("🔧 System Configuration")
    
    # System status
    st.sidebar.markdown("### System Status")
    monitor, kb, workflow = initialize_system()
    
    if monitor:
        st.sidebar.success("✅ LangSmith Monitoring: Active")
    else:
        st.sidebar.error("❌ LangSmith Monitoring: Inactive")
    
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

def render_analysis_progress(agent_status):
    """Render real-time analysis progress"""
    st.markdown("## 🔄 Analysis Progress")
    
    agents = [
        ("Research Agent", "🔬", "Gathering market and clinical data"),
        ("Competitive Analyst", "🏁", "Analyzing competitive landscape"), 
        ("Regulatory Specialist", "📋", "Assessing regulatory requirements"),
        ("Clinical Strategist", "🧪", "Developing clinical strategy"),
        ("Market Strategist", "📈", "Creating launch strategy"),
        ("Report Writer", "📝", "Compiling final report")
    ]
    
    progress_container = st.container()
    
    with progress_container:
        for i, (name, icon, description) in enumerate(agents):
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"**{icon} {name}**")
            
            with col2:
                if i < agent_status.get("current_step", 0):
                    st.success(f"✅ Complete: {description}")
                elif i == agent_status.get("current_step", 0):
                    st.info(f"🔄 In Progress: {description}")
                else:
                    st.markdown(f"⏳ Pending: {description}")
            
            with col3:
                if i < agent_status.get("current_step", 0):
                    st.markdown("✅")
                elif i == agent_status.get("current_step", 0):
                    st.markdown("🔄")
                else:
                    st.markdown("⏳")

def run_analysis(product_info, config):
    """Run the pharmaceutical launch analysis"""
    
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
        
        # Simulate progress updates (in real implementation, this would be integrated with the workflow)
        result = workflow.invoke(initial_state)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return result
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

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
        "🏁 Competitive Analysis", 
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
            if insights.get("timeline_estimates"):
                st.markdown("**Timeline Estimates:**")
                # Create a simple timeline chart
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

def render_monitoring_dashboard(monitor):
    """Render LangSmith monitoring dashboard"""
    st.markdown("## 📈 System Performance Dashboard")
    
    if not monitor:
        st.warning("Monitoring is not available. Please configure LangSmith credentials.")
        return
    
    try:
        dashboard_data = monitor.get_performance_dashboard()
        
        # Overall metrics
        st.markdown("### Overall Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Runs", dashboard_data["overall_metrics"]["total_runs"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True) 
            st.metric("Success Rate", f"{dashboard_data['overall_metrics']['success_rate']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Tokens", f"{dashboard_data['overall_metrics']['total_tokens_used']:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Est. Cost", f"${dashboard_data['overall_metrics']['estimated_cost']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Agent performance
        st.markdown("### Agent Performance")
        if dashboard_data["agent_performance"]:
            agent_df = pd.DataFrame.from_dict(dashboard_data["agent_performance"], orient='index')
            agent_df = agent_df.reset_index()
            agent_df.rename(columns={'index': 'Agent'}, inplace=True)
            
            # Success rate chart
            if 'total_executions' in agent_df.columns and 'successful_executions' in agent_df.columns:
                agent_df['success_rate'] = (agent_df['successful_executions'] / agent_df['total_executions'] * 100).fillna(0)
                
                fig = px.bar(agent_df, x='Agent', y='success_rate', 
                           title='Agent Success Rates',
                           color='success_rate',
                           color_continuous_scale='RdYlGn')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.dataframe(agent_df, use_container_width=True)
        else:
            st.info("No agent performance data available yet. Run an analysis to see metrics.")
        
        # Recommendations
        st.markdown("### Optimization Recommendations")
        recommendations = dashboard_data.get("recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
            
    except Exception as e:
        st.error(f"Error loading monitoring dashboard: {e}")

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
    
    # Render header
    render_header()
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    # Main content area
    if not st.session_state.analysis_running:
        # Product input form
        product_info = render_product_input()
        
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
        
        # Run the analysis
        product_info = {
            "product_name": st.session_state.get("product_name", "Test Product"),
            "drug_class": st.session_state.get("drug_class", "Test Class"),
            "indication": st.session_state.get("indication", "Test Indication")
        }
        
        with st.spinner("Running comprehensive pharmaceutical launch analysis..."):
            # Simulate analysis progress
            progress_container = st.container()
            
            with progress_container:
                agents_progress = [
                    "🔬 Research Agent: Gathering data...",
                    "🏁 Competitive Analyst: Analyzing competitors...", 
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
                
                # Mock result for demonstration
                mock_result = {
                    "product_name": product_info["product_name"],
                    "final_report": f"""
# Pharmaceutical Launch Analysis Report

## EXECUTIVE SUMMARY

Analysis completed for {product_info['product_name']} ({product_info['drug_class']}) targeting {product_info['indication']}.

**Key Findings:**
- Strong market opportunity identified in {product_info['indication']} treatment
- Competitive landscape shows room for differentiation
- Regulatory pathway is well-established for {product_info['drug_class']}
- Clinical development strategy should focus on superiority trials

**Recommendations:**
1. Proceed with Phase III development focusing on primary efficacy endpoint
2. Develop premium pricing strategy based on clinical differentiation
3. Establish early market access discussions with key payers
4. Build KOL relationships in target therapeutic area

## Investment Requirements: $150-200M
## Timeline to Launch: 18-24 months
## Peak Sales Potential: $500M-1B annually
                    """,
                    "research_findings": {
                        "market_analysis": {
                            "query": "What is the market opportunity?",
                            "insights": f"The {product_info['indication']} market shows strong growth potential with significant unmet medical needs.",
                            "sources": ["Market research reports", "Clinical publications"]
                        }
                    },
                    "competitive_analysis": {
                        "strategic_recommendations": f"Position {product_info['product_name']} as a best-in-class {product_info['drug_class']} with superior efficacy and safety profile."
                    },
                    "regulatory_assessment": {
                        "regulatory_strategy": f"FDA approval pathway for {product_info['drug_class']} is well-established. Recommend 505(b)(1) NDA submission.",
                        "approval_pathway": "Standard NDA",
                        "timeline_estimate": "12-18 months post-submission"
                    },
                    "clinical_insights": {
                        "development_strategy": f"Phase III program should demonstrate superiority over current standard of care in {product_info['indication']}."
                    },
                    "market_strategy": {
                        "comprehensive_strategy": f"Launch {product_info['product_name']} with premium pricing strategy, focusing on specialty care market initially."
                    }
                }
                
                st.session_state.analysis_result = mock_result
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
    
    # Monitoring dashboard (always visible in sidebar or separate section)
    if config["monitor"]:
        st.markdown("---")
        render_monitoring_dashboard(config["monitor"])

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
    
    ## Key Learning Points
    
    ### 1. LangChain Fundamentals
    ```python
    # Document ingestion and processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(documents, embeddings)
    
    # Custom RAG chains with domain-specific prompts
    prompt = PromptTemplate(template=regulatory_template)
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever, prompt=prompt)
    ```
    
    ### 2. LangGraph State Management
    ```python
    # Typed state definition
    class PharmaLaunchState(TypedDict):
        product_name: str
        research_findings: Dict[str, Any]
        competitive_analysis: Dict[str, Any]
        # ... more fields
    
    # Agent workflow definition
    workflow = StateGraph(PharmaLaunchState)
    workflow.add_node("research_agent", agents.research_agent)
    workflow.add_edge("research_agent", "competitive_analyst")
    ```
    
    ### 3. LangSmith Integration
    ```python
    # Environment setup for tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "your_key"
    
    # Custom evaluators
    def pharma_accuracy_evaluator(run, example):
        # Domain-specific evaluation logic
        return {"score": accuracy_score, "reason": feedback}
    ```
    
    ## Pharmaceutical Domain Expertise
    
    The system incorporates deep pharmaceutical knowledge:
    
    - **Regulatory Intelligence**: FDA guidance documents, approval pathways
    - **Clinical Development**: Trial design, endpoints, patient populations  
    - **Market Analysis**: Competitive landscape, pricing, market access
    - **Risk Assessment**: Development risks, commercial risks, mitigation strategies
    
    ## Next Steps for Learning
    
    1. **Experiment** with different drug classes and indications
    2. **Customize** agent prompts for your specific use cases
    3. **Extend** with additional data sources (patents, publications)
    4. **Optimize** using LangSmith insights and evaluation results
    5. **Deploy** to production with proper scaling and monitoring
    
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