# langgraph_agents.py
from typing import Dict, List, Any, TypedDict, Annotated
import json
from dataclasses import dataclass

from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.schema import HumanMessage, AIMessage

from langchain_foundation import PharmaKnowledgeBase, PharmaRAGChains
from config import Config

# State definition for the multi-agent workflow
class PharmaLaunchState(TypedDict):
    product_name: str
    indication: str
    drug_class: str
    current_agent: str
    research_findings: Dict[str, Any]
    competitive_analysis: Dict[str, Any]
    regulatory_assessment: Dict[str, Any]
    clinical_insights: Dict[str, Any]
    market_strategy: Dict[str, Any]
    final_report: str
    iteration_count: int
    messages: List[dict]

class PharmaLaunchAgents:
    """Multi-agent system for pharmaceutical product launch analysis"""
    
    def __init__(self, knowledge_base: PharmaKnowledgeBase):
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0.1)
        self.kb = knowledge_base
        self.chains = PharmaRAGChains(knowledge_base)
        
        # Initialize specialized chains
        self.regulatory_chain = self.chains.create_regulatory_chain()
        self.competitive_chain = self.chains.create_competitive_analysis_chain()
        self.clinical_chain = self.chains.create_clinical_development_chain()
        
    def research_agent(self, state: PharmaLaunchState) -> PharmaLaunchState:
        """Agent responsible for gathering comprehensive research data"""
        print(f"🔬 Research Agent: Analyzing {state['product_name']} for {state['indication']}")
        
        research_queries = [
            f"What is known about {state['drug_class']} drugs for {state['indication']}?",
            f"What are current treatment options for {state['indication']}?",
            f"What are the unmet medical needs in {state['indication']}?",
            f"What are recent developments in {state['indication']} treatment?"
        ]
        
        findings = {}
        for i, query in enumerate(research_queries):
            try:
                # Use the knowledge base to answer research questions
                result = self.kb.vector_store.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in result])
                
                # Generate insights using LLM
                prompt = ChatPromptTemplate.from_template(
                    "Based on the following context about {indication} and {drug_class}, provide key insights for: {query}\n\nContext: {context}\n\nInsights:"
                )
                
                response = self.llm.invoke(
                    prompt.format(
                        indication=state['indication'],
                        drug_class=state['drug_class'],
                        query=query,
                        context=context
                    )
                )
                
                findings[f"research_point_{i+1}"] = {
                    "query": query,
                    "insights": response.content,
                    "sources": [doc.metadata.get('source', 'Unknown') for doc in result]
                }
            except Exception as e:
                findings[f"research_point_{i+1}"] = {
                    "query": query,
                    "insights": f"Research limited due to data availability: {str(e)}",
                    "sources": []
                }
        
        state["research_findings"] = findings
        state["current_agent"] = "competitive_analyst"
        state["messages"].append({
            "agent": "research_agent",
            "message": f"Completed research analysis with {len(findings)} key findings",
            "timestamp": "now"
        })
        
        return state
    
    def competitive_analyst(self, state: PharmaLaunchState) -> PharmaLaunchState:
        """Agent focused on competitive landscape analysis"""
        print(f"🏁 Competitive Analyst: Evaluating market position for {state['product_name']}")
        
        try:
            competitive_queries = [
                f"What are the competing drugs for {state['indication']}?",
                f"What is the market share distribution in {state['indication']}?",
                f"What are the pricing strategies for {state['drug_class']} drugs?",
                f"What are the key differentiators in the {state['indication']} market?"
            ]
            
            analysis = {}
            for query in competitive_queries:
                result = self.competitive_chain.run(query)
                analysis[query] = result
            
            # Synthesize competitive positioning
            synthesis_prompt = f"""
            Based on the competitive analysis for {state['product_name']} ({state['drug_class']}) for {state['indication']}, 
            provide strategic recommendations for market positioning, pricing, and differentiation.
            
            Analysis Results:
            {json.dumps(analysis, indent=2)}
            
            Provide structured recommendations covering:
            1. Competitive positioning strategy
            2. Pricing considerations
            3. Key differentiators to emphasize  
            4. Market entry strategy
            5. Competitive risks and mitigation
            """
            
            recommendations = self.llm.invoke(synthesis_prompt)
            
            state["competitive_analysis"] = {
                "detailed_analysis": analysis,
                "strategic_recommendations": recommendations.content,
                "key_competitors": "Analysis based on available data",
                "market_opportunities": "Identified through research"
            }
            
        except Exception as e:
            state["competitive_analysis"] = {
                "error": f"Competitive analysis limited: {str(e)}",
                "recommendations": "Conduct additional market research for comprehensive competitive intelligence"
            }
        
        state["current_agent"] = "regulatory_specialist"
        state["messages"].append({
            "agent": "competitive_analyst", 
            "message": "Completed competitive landscape analysis",
            "timestamp": "now"
        })
        
        return state
    
    def regulatory_specialist(self, state: PharmaLaunchState) -> PharmaLaunchState:
        """Agent specialized in regulatory requirements and compliance"""
        print(f"📋 Regulatory Specialist: Assessing requirements for {state['product_name']}")
        
        try:
            regulatory_queries = [
                f"What are FDA approval requirements for {state['drug_class']} drugs?",
                f"What regulatory pathway is appropriate for {state['indication']} treatment?",
                f"What post-marketing requirements apply to {state['drug_class']}?",
                f"What are key regulatory risks for {state['indication']} drugs?"
            ]
            
            assessment = {}
            for query in regulatory_queries:
                result = self.regulatory_chain.run(query)
                assessment[query] = result
            
            # Create regulatory timeline and strategy
            timeline_prompt = f"""
            Based on regulatory assessment for {state['product_name']} ({state['drug_class']}) for {state['indication']},
            create a regulatory strategy and timeline.
            
            Assessment Results:
            {json.dumps(assessment, indent=2)}
            
            Provide:
            1. Recommended regulatory pathway
            2. Key regulatory milestones and timeline
            3. Required studies and documentation
            4. Risk mitigation strategies
            5. Regulatory agency interaction plan
            """
            
            strategy = self.llm.invoke(timeline_prompt)
            
            state["regulatory_assessment"] = {
                "detailed_assessment": assessment,
                "regulatory_strategy": strategy.content,
                "approval_pathway": "Based on drug class and indication",
                "timeline_estimate": "Detailed in strategy section"
            }
            
        except Exception as e:
            state["regulatory_assessment"] = {
                "error": f"Regulatory assessment limited: {str(e)}",
                "recommendations": "Consult with regulatory experts for detailed pathway analysis"
            }
        
        state["current_agent"] = "clinical_strategist"
        state["messages"].append({
            "agent": "regulatory_specialist",
            "message": "Completed regulatory assessment and strategy development", 
            "timestamp": "now"
        })
        
        return state
    
    def clinical_strategist(self, state: PharmaLaunchState) -> PharmaLaunchState:
        """Agent focused on clinical development strategy"""
        print(f"🧪 Clinical Strategist: Developing clinical plan for {state['product_name']}")
        
        try:
            clinical_queries = [
                f"What clinical trial designs are optimal for {state['indication']}?",
                f"What are typical development timelines for {state['drug_class']}?", 
                f"What are key clinical endpoints for {state['indication']}?",
                f"What patient population considerations apply to {state['indication']}?"
            ]
            
            insights = {}
            for query in clinical_queries:
                result = self.clinical_chain.run(query)
                insights[query] = result
            
            # Develop clinical strategy
            strategy_prompt = f"""
            Based on clinical insights for {state['product_name']} ({state['drug_class']}) for {state['indication']},
            develop a comprehensive clinical development strategy.
            
            Clinical Insights:
            {json.dumps(insights, indent=2)}
            
            Research Findings Context:
            {json.dumps(state.get('research_findings', {}), indent=2)}
            
            Provide:
            1. Clinical development plan overview
            2. Phase-by-phase strategy and timeline
            3. Patient recruitment considerations
            4. Key success factors and risks
            5. Budget and resource estimates
            """
            
            development_plan = self.llm.invoke(strategy_prompt)
            
            state["clinical_insights"] = {
                "detailed_insights": insights,
                "development_strategy": development_plan.content,
                "timeline_estimates": "Included in development strategy",
                "success_factors": "Analysis based on indication requirements"
            }
            
        except Exception as e:
            state["clinical_insights"] = {
                "error": f"Clinical strategy development limited: {str(e)}",
                "recommendations": "Engage clinical development experts for detailed planning"
            }
        
        state["current_agent"] = "market_strategist"
        state["messages"].append({
            "agent": "clinical_strategist",
            "message": "Completed clinical development strategy",
            "timestamp": "now"
        })
        
        return state
    
    def market_strategist(self, state: PharmaLaunchState) -> PharmaLaunchState:
        """Agent responsible for overall market strategy and go-to-market planning"""
        print(f"📈 Market Strategist: Creating launch strategy for {state['product_name']}")
        
        # Synthesize all previous analyses into market strategy
        synthesis_prompt = f"""
        Create a comprehensive market launch strategy for {state['product_name']} ({state['drug_class']}) for {state['indication']}.
        
        Base your strategy on the following analyses:
        
        Research Findings:
        {json.dumps(state.get('research_findings', {}), indent=2)}
        
        Competitive Analysis:
        {json.dumps(state.get('competitive_analysis', {}), indent=2)}
        
        Regulatory Assessment: 
        {json.dumps(state.get('regulatory_assessment', {}), indent=2)}
        
        Clinical Insights:
        {json.dumps(state.get('clinical_insights', {}), indent=2)}
        
        Develop a comprehensive market strategy covering:
        1. Target market segmentation and sizing
        2. Value proposition and positioning
        3. Pricing and market access strategy
        4. Launch timeline and milestones
        5. Key success metrics and KPIs
        6. Risk assessment and mitigation plans
        7. Resource requirements and budget estimates
        """
        
        strategy = self.llm.invoke(synthesis_prompt)
        
        state["market_strategy"] = {
            "comprehensive_strategy": strategy.content,
            "launch_readiness": "Strategy developed based on available data",
            "next_steps": "Detailed implementation planning required"
        }
        
        state["current_agent"] = "report_writer"
        state["messages"].append({
            "agent": "market_strategist",
            "message": "Completed comprehensive market launch strategy",
            "timestamp": "now"
        })
        
        return state
    
    def report_writer(self, state: PharmaLaunchState) -> PharmaLaunchState:
        """Agent that creates the final comprehensive report"""
        print(f"📝 Report Writer: Compiling final report for {state['product_name']}")
        
        report_prompt = f"""
        Create a comprehensive executive summary and launch readiness report for {state['product_name']} ({state['drug_class']}) for {state['indication']}.
        
        Synthesize all analyses into a clear, actionable report:
        
        RESEARCH FINDINGS:
        {json.dumps(state.get('research_findings', {}), indent=2)}
        
        COMPETITIVE ANALYSIS:
        {json.dumps(state.get('competitive_analysis', {}), indent=2)}
        
        REGULATORY ASSESSMENT:
        {json.dumps(state.get('regulatory_assessment', {}), indent=2)}
        
        CLINICAL INSIGHTS:
        {json.dumps(state.get('clinical_insights', {}), indent=2)}
        
        MARKET STRATEGY:
        {json.dumps(state.get('market_strategy', {}), indent=2)}
        
        Format as a professional pharmaceutical business report with:
        
        EXECUTIVE SUMMARY
        - Key findings and recommendations
        - Investment requirements and timeline
        - Expected ROI and market potential
        
        DETAILED ANALYSIS
        - Market opportunity assessment
        - Competitive landscape
        - Regulatory pathway and timeline
        - Clinical development strategy
        - Commercial strategy and launch plan
        
        RISK ASSESSMENT
        - Key risks and mitigation strategies
        - Success probability assessment
        
        RECOMMENDATIONS & NEXT STEPS
        - Immediate action items
        - Resource allocation recommendations
        - Decision points and milestones
        """
        
        final_report = self.llm.invoke(report_prompt)
        
        state["final_report"] = final_report.content
        state["current_agent"] = "complete"
        state["messages"].append({
            "agent": "report_writer",
            "message": f"Final comprehensive report completed for {state['product_name']}",
            "timestamp": "now"  
        })
        
        return state

def create_pharma_workflow(knowledge_base: PharmaKnowledgeBase) -> StateGraph:
    """Create the multi-agent workflow graph"""
    
    agents = PharmaLaunchAgents(knowledge_base)
    
    # Create the state graph
    workflow = StateGraph(PharmaLaunchState)
    
    # Add agents as nodes
    workflow.add_node("research_agent", agents.research_agent)
    workflow.add_node("competitive_analyst", agents.competitive_analyst) 
    workflow.add_node("regulatory_specialist", agents.regulatory_specialist)
    workflow.add_node("clinical_strategist", agents.clinical_strategist)
    workflow.add_node("market_strategist", agents.market_strategist)
    workflow.add_node("report_writer", agents.report_writer)
    
    # Define the workflow edges
    workflow.add_edge("research_agent", "competitive_analyst")
    workflow.add_edge("competitive_analyst", "regulatory_specialist")
    workflow.add_edge("regulatory_specialist", "clinical_strategist")
    workflow.add_edge("clinical_strategist", "market_strategist")
    workflow.add_edge("market_strategist", "report_writer")
    workflow.add_edge("report_writer", END)
    
    # Set entry point
    workflow.set_entry_point("research_agent")
    
    return workflow.compile()

# Usage example
if __name__ == "__main__":
    from langchain_foundation import PharmaKnowledgeBase
    
    # Initialize knowledge base
    print("Building knowledge base...")
    kb = PharmaKnowledgeBase()
    kb.build_knowledge_base("immunotherapy", "cancer")
    
    # Create workflow
    print("Creating multi-agent workflow...")
    workflow = create_pharma_workflow(kb)
    
    # Initial state
    initial_state = PharmaLaunchState(
        product_name="ImmunoMax",
        indication="advanced melanoma", 
        drug_class="PD-1 inhibitor",
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
    
    # Run the workflow
    print("Running pharma launch analysis workflow...")
    result = workflow.invoke(initial_state)
    
    print("\n" + "="*50)
    print("PHARMACEUTICAL LAUNCH ANALYSIS COMPLETE")
    print("="*50)
    print(result["final_report"])