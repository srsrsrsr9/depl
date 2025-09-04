# 🚀 Pharmaceutical Launch Intelligence System - Setup Guide

## Overview
This project demonstrates the power of the LangChain ecosystem (LangChain, LangGraph, LangSmith) through a comprehensive pharmaceutical product launch analysis system. You'll learn how to build multi-agent workflows, implement domain-specific RAG systems, and monitor AI applications in production.

## 📋 Prerequisites

### Required Accounts & API Keys
1. **OpenAI Account** - For LLM access
   - Sign up at https://platform.openai.com
   - Generate API key from API section
   
2. **LangSmith Account** - For monitoring and tracing  
   - Sign up at https://smith.langchain.com
   - Create project and get API key

### System Requirements
- Python 3.8+
- 4GB+ RAM (for vector embeddings)
- Stable internet connection

## 🛠️ Installation Steps

### 1. Clone and Setup Environment

```bash
# Create project directory
mkdir pharma-launch-ai
cd pharma-launch-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith Configuration  
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=pharma-launch-assistant

# Optional: Custom configurations
VECTOR_STORE_PATH=./data/vector_store
```

### 3. Test Basic Setup

```bash
# Test LangChain foundation
python langchain_foundation.py

# Test LangGraph agents  
python langgraph_agents.py

# Test LangSmith monitoring
python langsmith_monitoring.py
```

### 4. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📚 Learning Path

### Phase 1: LangChain Fundamentals (Week 1)
**Focus**: Document processing, embeddings, RAG chains

**Key Concepts:**
- Vector embeddings and similarity search
- Document loaders and text splitters
- Custom prompt templates
- Retrieval-Augmented Generation (RAG)

**Hands-on Tasks:**
1. Run the knowledge base builder with different drug classes
2. Experiment with different chunk sizes and overlap settings
3. Create custom prompts for different query types
4. Test the RAG chains with various pharmaceutical questions

**Code to Study:**
```python
# langchain_foundation.py - Focus on:
class PharmaDataIngestion  # Document processing
class PharmaKnowledgeBase  # Vector storage  
class PharmaRAGChains     # Custom chains
```

### Phase 2: LangGraph Multi-Agent Systems (Week 2)
**Focus**: Workflow orchestration, state management, agent coordination

**Key Concepts:**
- StateGraph and typed state definitions
- Node-based agent architecture
- Sequential and conditional workflows
- Error handling and recovery

**Hands-on Tasks:**
1. Modify agent prompts and observe behavior changes
2. Add a new agent to the workflow (e.g., Patent Analysis Agent)
3. Implement conditional routing between agents
4. Add error recovery mechanisms

**Code to Study:**
```python
# langgraph_agents.py - Focus on:
class PharmaLaunchState    # State management
class PharmaLaunchAgents   # Individual agents
def create_pharma_workflow # Workflow definition
```

### Phase 3: LangSmith Monitoring & Optimization (Week 3)
**Focus**: Observability, evaluation, performance optimization

**Key Concepts:**
- Distributed tracing for LLM applications
- Custom evaluation metrics
- Performance monitoring and cost tracking
- A/B testing and optimization

**Hands-on Tasks:**
1. Create custom evaluators for your domain
2. Set up evaluation datasets
3. Monitor token usage and optimize costs
4. Implement performance alerts

**Code to Study:**
```python
# langsmith_monitoring.py - Focus on:
class PharmaLangSmithMonitor  # Monitoring system
def pharma_accuracy_evaluator # Custom evaluation
def run_evaluation           # Evaluation pipeline
```

## 🎯 Key Learning Outcomes

By completing this project, you'll master:

### LangChain Skills
- ✅ Document ingestion from multiple sources
- ✅ Vector embeddings and similarity search
- ✅ Custom RAG chain development
- ✅ Domain-specific prompt engineering
- ✅ Tool integration and function calling

### LangGraph Skills  
- ✅ Multi-agent system architecture
- ✅ State management across agents
- ✅ Workflow orchestration and routing
- ✅ Error handling and recovery patterns
- ✅ Complex business process automation

### LangSmith Skills
- ✅ End-to-end tracing and monitoring
- ✅ Custom evaluation framework development
- ✅ Performance optimization techniques
- ✅ Production monitoring and alerting
- ✅ Cost tracking and optimization

## 🚀 Advanced Extensions

Once you've mastered the basics, try these extensions:

### 1. Data Source Integration
- **Patent Data**: USPTO API integration
- **Scientific Literature**: PubMed API
- **Market Data**: Bloomberg/Reuters APIs
- **Real-time News**: News API integration

### 2. Advanced AI Capabilities
- **Image Analysis**: FDA label analysis with vision models
- **Time Series**: Market forecasting with specialized models
- **Knowledge Graphs**: Entity relationship mapping
- **Multi-modal**: Combine text, images, and structured data

### 3. Production Features
- **Authentication**: User management and access control
- **Caching**: Redis integration for performance
- **Scaling**: Celery for background processing
- **Deployment**: Docker containers and cloud deployment

### 4. Domain Expansion
- **Medical Devices**: FDA 510(k) pathway analysis
- **Biotechnology**: Gene therapy regulatory requirements
- **Digital Health**: Software as medical device (SaMD)
- **International**: EMA, PMDA regulatory pathways

## 🔧 Troubleshooting

### Common Issues

**1. API Key Issues**
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $LANGCHAIN_API_KEY

# Verify .env file loading
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY')[:10])"
```

**2. Vector Store Issues**
```bash
# Clear and rebuild vector store
rm -rf ./data/vector_store
python langchain_foundation.py
```

**3. Memory Issues**
```bash
# Reduce chunk size in config.py
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
```

**4. Network Issues**
```bash
# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

### Getting Help

1. **Documentation**: Check the official docs
   - LangChain: https://docs.langchain.com
   - LangSmith: https://docs.smith.langchain.com

2. **Community**: Join discussions
   - LangChain Discord: https://discord.gg/langchain
   - GitHub Issues: Report bugs and feature requests

3. **Examples**: Study additional examples
   - LangChain Templates: https://github.com/langchain-ai/langchain/tree/master/templates
   - LangGraph Examples: https://github.com/langchain-ai/langgraph/tree/main/examples

## 📈 Next Steps

1. **Complete the Tutorial**: Work through all three phases systematically
2. **Build Your Own**: Apply the patterns to your domain
3. **Contribute**: Share improvements and extensions
4. **Scale**: Deploy to production with monitoring
5. **Explore**: Dive deeper into advanced LangChain features

## 💡 Pro Tips

1. **Start Simple**: Begin with basic RAG before adding agents
2. **Monitor Early**: Set up LangSmith from the beginning  
3. **Iterate Fast**: Use the evaluation framework for rapid improvement
4. **Think Modular**: Design reusable components and patterns
5. **Document Everything**: Good documentation accelerates learning

---

**Happy Learning! 🎓**

This project provides a solid foundation for building production-grade AI applications with the LangChain ecosystem. Take your time with each phase, experiment freely, and don't hesitate to modify the code to match your specific interests and use cases.