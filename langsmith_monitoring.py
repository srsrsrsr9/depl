# langsmith_monitoring.py
import os
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import wraps

from langsmith import Client
from langsmith.schemas import Run, Example
# from langsmith.evaluation import LangChainStringEvaluator #, evaluate
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks import StdOutCallbackHandler
from langchain.schema import LLMResult

from config import Config

class PharmaLangSmithMonitor:
    """Comprehensive monitoring and analytics for the pharma launch system"""
    
    def __init__(self):
        # Initialize LangSmith client
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = Config.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = Config.LANGCHAIN_PROJECT
        
        self.client = Client()
        self.tracer = LangChainTracer(project_name=Config.LANGCHAIN_PROJECT)
        
        # Metrics tracking
        self.metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "average_duration": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "agent_performance": {}
        }
        
    def create_dataset(self, name: str, description: str = ""):
        """Create a dataset for evaluation"""
        try:
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=description
            )
            print(f"Created dataset: {name}")
            return dataset
        except Exception as e:
            print(f"Dataset might already exist or error occurred: {e}")
            return self.client.read_dataset(dataset_name=name)
    
    def add_examples_to_dataset(self, dataset_name: str, examples: List[Dict]):
        """Add evaluation examples to dataset"""
        dataset = self.client.read_dataset(dataset_name=dataset_name)
        
        for example in examples:
            self.client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                dataset_id=dataset.id
            )
        print(f"Added {len(examples)} examples to dataset {dataset_name}")
    
    def pharma_accuracy_evaluator(self):
        """Custom evaluator for pharmaceutical domain accuracy"""
        
        def evaluate_accuracy(run: Run, example: Example) -> Dict:
            """Evaluate accuracy of pharma-specific outputs"""
            
            # Extract actual and expected outputs
            prediction = run.outputs.get("output", "") if run.outputs else ""
            reference = example.outputs.get("output", "") if example.outputs else ""
            
            # Pharma-specific accuracy checks
            accuracy_score = 0
            feedback = []
            
            # Check for regulatory compliance mentions
            regulatory_keywords = ["FDA", "regulatory", "compliance", "approval", "guidance"]
            if any(keyword.lower() in prediction.lower() for keyword in regulatory_keywords):
                accuracy_score += 0.2
                feedback.append("Mentions regulatory considerations")
            
            # Check for competitive analysis
            competitive_keywords = ["competitor", "market share", "positioning", "differentiation"]
            if any(keyword.lower() in prediction.lower() for keyword in competitive_keywords):
                accuracy_score += 0.2
                feedback.append("Includes competitive analysis")
            
            # Check for clinical considerations
            clinical_keywords = ["clinical trial", "efficacy", "safety", "patient", "endpoint"]
            if any(keyword.lower() in prediction.lower() for keyword in clinical_keywords):
                accuracy_score += 0.2
                feedback.append("Addresses clinical aspects")
            
            # Check for market strategy elements
            strategy_keywords = ["pricing", "launch", "market access", "timeline", "ROI"]
            if any(keyword.lower() in prediction.lower() for keyword in strategy_keywords):
                accuracy_score += 0.2
                feedback.append("Contains market strategy elements")
            
            # Check for risk assessment
            risk_keywords = ["risk", "mitigation", "challenge", "probability"]
            if any(keyword.lower() in prediction.lower() for keyword in risk_keywords):
                accuracy_score += 0.2
                feedback.append("Includes risk assessment")
            
            return {
                "key": "pharma_accuracy",
                "score": accuracy_score,
                "reason": "; ".join(feedback) if feedback else "Limited pharma domain coverage"
            }
        
        return evaluate_accuracy
    
    def completeness_evaluator(self):
        """Evaluate completeness of pharma launch analysis"""
        
        def evaluate_completeness(run: Run, example: Example) -> Dict:
            prediction = run.outputs.get("output", "") if run.outputs else ""
            
            # Required sections for pharma launch analysis
            required_sections = {
                "executive_summary": ["executive", "summary", "overview"],
                "market_analysis": ["market", "opportunity", "size"],
                "competitive_landscape": ["competitor", "competitive", "landscape"],
                "regulatory_pathway": ["regulatory", "FDA", "approval"],
                "clinical_strategy": ["clinical", "trial", "development"],
                "financial_projections": ["financial", "revenue", "cost", "ROI"],
                "risk_assessment": ["risk", "mitigation", "challenge"],
                "recommendations": ["recommend", "next steps", "action"]
            }
            
            found_sections = 0
            missing_sections = []
            
            for section, keywords in required_sections.items():
                if any(keyword.lower() in prediction.lower() for keyword in keywords):
                    found_sections += 1
                else:
                    missing_sections.append(section)
            
            completeness_score = found_sections / len(required_sections)
            
            return {
                "key": "completeness",
                "score": completeness_score,
                "reason": f"Found {found_sections}/{len(required_sections)} sections. Missing: {', '.join(missing_sections)}"
            }
        
        return evaluate_completeness
    
    def create_evaluation_examples(self) -> List[Dict]:
        """Create evaluation examples for pharma launch analysis"""
        
        examples = [
            {
                "inputs": {
                    "product_name": "TestDrug1",
                    "indication": "diabetes",
                    "drug_class": "GLP-1 agonist"
                },
                "outputs": {
                    "output": "Complete analysis should include regulatory pathway, competitive landscape, clinical development strategy, market positioning, pricing strategy, and risk assessment with specific recommendations for launch success."
                }
            },
            {
                "inputs": {
                    "product_name": "TestDrug2", 
                    "indication": "hypertension",
                    "drug_class": "ACE inhibitor"
                },
                "outputs": {
                    "output": "Analysis must cover FDA approval requirements, competitive differentiation from existing ACE inhibitors, clinical trial design considerations, market access strategy, and financial projections with timeline milestones."
                }
            },
            {
                "inputs": {
                    "product_name": "TestDrug3",
                    "indication": "depression", 
                    "drug_class": "SSRI"
                },
                "outputs": {
                    "output": "Comprehensive evaluation including regulatory compliance for CNS drugs, competitive positioning against established SSRIs, clinical endpoints for depression, market segmentation, and risk mitigation strategies."
                }
            }
        ]
        
        return examples
    
    # def run_evaluation(self, workflow_function, dataset_name: str = "pharma_launch_eval"):
    #     """Run comprehensive evaluation of the pharma workflow"""
        
    #     # Create dataset if it doesn't exist
    #     try:
    #         dataset = self.create_dataset(
    #             name=dataset_name,
    #             description="Evaluation dataset for pharmaceutical product launch analysis"
    #         )
            
    #         # Add examples to dataset
    #         examples = self.create_evaluation_examples()
    #         self.add_examples_to_dataset(dataset_name, examples)
            
    #     except Exception as e:
    #         print(f"Using existing dataset: {e}")
        
    #     # Define evaluation configuration
    #     def workflow_wrapper(inputs: Dict) -> Dict:
    #         """Wrapper for the workflow function to match evaluator interface"""
    #         try:
    #             result = workflow_function(inputs)
    #             return {"output": result.get("final_report", "No output generated")}
    #         except Exception as e:
    #             return {"output": f"Error: {str(e)}"}
        
    #     # Run evaluation
    #     evaluation_results = evaluate(
    #         workflow_wrapper,
    #         data=dataset_name,
    #         evaluators=[
    #             self.pharma_accuracy_evaluator(),
    #             self.completeness_evaluator(),
    #             LangChainStringEvaluator("criteria", 
    #                 criteria={
    #                     "coherence": "Is the analysis logically structured and coherent?",
    #                     "relevance": "Is the content relevant to pharmaceutical product launch?",
    #                     "actionability": "Does it provide actionable recommendations?"
    #                 })
    #         ],
    #         project_name=f"{Config.LANGCHAIN_PROJECT}_evaluation"
    #     )
        
    #     return evaluation_results
    
    def track_agent_performance(self, agent_name: str, execution_time: float, 
                              success: bool, tokens_used: int = 0):
        """Track individual agent performance metrics"""
        
        if agent_name not in self.metrics["agent_performance"]:
            self.metrics["agent_performance"][agent_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0,
                "total_tokens": 0,
                "average_time": 0
            }
        
        agent_metrics = self.metrics["agent_performance"][agent_name]
        agent_metrics["total_executions"] += 1
        agent_metrics["total_time"] += execution_time
        agent_metrics["total_tokens"] += tokens_used
        agent_metrics["average_time"] = agent_metrics["total_time"] / agent_metrics["total_executions"]
        
        if success:
            agent_metrics["successful_executions"] += 1
            self.metrics["successful_runs"] += 1
        else:
            self.metrics["failed_runs"] += 1
        
        self.metrics["total_runs"] += 1
        self.metrics["total_tokens"] += tokens_used
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Generate performance dashboard data"""
        
        dashboard_data = {
            "overall_metrics": {
                "total_runs": self.metrics["total_runs"],
                "success_rate": (self.metrics["successful_runs"] / max(self.metrics["total_runs"], 1)) * 100,
                "total_tokens_used": self.metrics["total_tokens"],
                "estimated_cost": self.metrics["total_tokens"] * 0.00002  # Rough estimate
            },
            "agent_performance": self.metrics["agent_performance"],
            "recommendations": self._generate_optimization_recommendations()
        }
        
        return dashboard_data
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        
        recommendations = []
        
        if self.metrics["total_runs"] > 0:
            success_rate = (self.metrics["successful_runs"] / self.metrics["total_runs"]) * 100
            
            if success_rate < 90:
                recommendations.append("Consider improving error handling and fallback mechanisms")
            
            if self.metrics["total_tokens"] > 100000:
                recommendations.append("Optimize prompts to reduce token usage and costs")
            
            # Analyze agent performance
            for agent, metrics in self.metrics["agent_performance"].items():
                if metrics["total_executions"] > 0:
                    agent_success_rate = (metrics["successful_executions"] / metrics["total_executions"]) * 100
                    if agent_success_rate < 85:
                        recommendations.append(f"Improve {agent} reliability - current success rate: {agent_success_rate:.1f}%")
                    
                    if metrics["average_time"] > 30:  # seconds
                        recommendations.append(f"Optimize {agent} performance - average execution time: {metrics['average_time']:.1f}s")
        
        if not recommendations:
            recommendations.append("System performing well - continue monitoring")
        
        return recommendations
    
    def export_metrics(self, filepath: str = "pharma_metrics.json"):
        """Export metrics to JSON file"""
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "dashboard": self.get_performance_dashboard()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Metrics exported to {filepath}")

def monitor_workflow_decorator(monitor: PharmaLangSmithMonitor):
    """Decorator to add monitoring to workflow functions"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            agent_name = func.__name__
            success = False
            
            try:
                # Execute function with tracing
                result = func(*args, **kwargs)
                success = True
                return result
                
            except Exception as e:
                print(f"Error in {agent_name}: {e}")
                raise
                
            finally:
                execution_time = time.time() - start_time
                # Note: Token counting would need to be implemented based on actual LLM calls
                monitor.track_agent_performance(agent_name, execution_time, success, tokens_used=0)
        
        return wrapper
    return decorator

# Enhanced workflow integration
def create_monitored_pharma_workflow(knowledge_base, monitor: PharmaLangSmithMonitor):
    """Create workflow with integrated monitoring"""
    
    from langgraph_agents import PharmaLaunchAgents, create_pharma_workflow
    
    # Apply monitoring decorators to agents
    agents = PharmaLaunchAgents(knowledge_base)
    
    # Wrap agent methods with monitoring
    agents.research_agent = monitor_workflow_decorator(monitor)(agents.research_agent)
    agents.competitive_analyst = monitor_workflow_decorator(monitor)(agents.competitive_analyst)
    agents.regulatory_specialist = monitor_workflow_decorator(monitor)(agents.regulatory_specialist)
    agents.clinical_strategist = monitor_workflow_decorator(monitor)(agents.clinical_strategist)
    agents.market_strategist = monitor_workflow_decorator(monitor)(agents.market_strategist)
    agents.report_writer = monitor_workflow_decorator(monitor)(agents.report_writer)
    
    # Create workflow with monitoring
    return create_pharma_workflow(knowledge_base)

# Usage example
if __name__ == "__main__":
    from langchain_foundation import PharmaKnowledgeBase
    
    # Initialize monitoring
    monitor = PharmaLangSmithMonitor()
    
    # Initialize knowledge base
    kb = PharmaKnowledgeBase()
    kb.build_knowledge_base("oncology", "cancer")
    
    # Create monitored workflow
    workflow = create_monitored_pharma_workflow(kb, monitor)
    
    # Example evaluation run
    def test_workflow(inputs):
        """Test function for evaluation"""
        from langgraph_agents import PharmaLaunchState
        
        initial_state = PharmaLaunchState(
            product_name=inputs["product_name"],
            indication=inputs["indication"], 
            drug_class=inputs["drug_class"],
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
        
        result = workflow.invoke(initial_state)
        return result
    
    # Run evaluation
    # print("Running evaluation...")
    # eval_results = monitor.run_evaluation(test_workflow)
    
    # Generate performance dashboard
    dashboard = monitor.get_performance_dashboard()
    print("\nPerformance Dashboard:")
    print(json.dumps(dashboard, indent=2))
    
    # Export metrics
    monitor.export_metrics()