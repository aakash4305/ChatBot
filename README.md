# MedicalQA Chatbot

A Q&A chatbot prototype designed to answer questions related to medical publications, specifically targeting breast cancer research with a focus on HER-2/neu oncogene amplification.

## 🔍 Overview

This chatbot prototype was developed to demonstrate an intelligent question-answering system that can effectively process and respond to queries about specific medical research publications. The primary focus is on the publication "[Human breast cancer: correlation of relapse and survival with amplification of the HER-2/neu oncogene](https://www.researchgate.net/profile/Gary-Clark/publication/19364043_Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_breast_cancer_correlation_of_relapse_and_survival_with_amplification_of_the_HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/0046352b85f241a532000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-McGuire-WLHuman-breast-cancer-correlation-of-relapse-and-survival-with-amplification-of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf)."

The project focuses on effective document-based question answering using RAG (Retrieval-Augmented Generation) technology and follows a comprehensive prototype development framework with defined evaluation metrics.

## 🚀 Development Framework

### Prototype Development

- **Technology Stack**: Python-based implementation using open-source Large Language Models
- **Core Functionality**: 
  - Question answering about breast cancer research related to HER-2/neu oncogene
  - Context-aware responses based on provided research publications
  - Effective document chunking and embedding for accurate information retrieval
  - Handles a wide range of questions with contextually relevant answers

### Evaluation Approach

- **Business Metrics**: Comprehensive KPIs including:
  - Response accuracy measurement
  - User satisfaction tracking
  - Response time monitoring
- **Testing Protocol**: Extensive testing with diverse query sets
- **Continuous Improvement**: Feedback collection and performance analysis pipeline

### Key Assumptions

- **Data Availability**: Full access to publication content
- **User Diversity**: Support for users with varying familiarity with the research material

## 🛠️ Technologies Used

- Python 3.8+
- Open-source LLM integration
- LangChain framework for LLM orchestration
- PyPDF for document processing
- Milvus for vector database storage
- SentenceTransformer for text embeddings
- Gradio for the user interface

## 📋 Prerequisites

To run this chatbot, you need:

- Python 3.8 or higher
- Git
- 8GB RAM (minimum)
- Internet connection (for model downloads if applicable)

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/aakash4305/ChatBot.git
cd ChatBot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the necessary research papers and place them in the `data` directory.

## 🏃 Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501).

3. Enter your questions about the research paper in the input field and receive comprehensive answers.

## 📊 Detailed Evaluation System

The chatbot implements a comprehensive evaluation system based on key business metrics:

### Performance Metrics

- **Response Accuracy**: 
  - LLM-based evaluation comparing responses against ground truth
  - Factual correctness measurement
  - Support for manual verification

- **Response Quality**:
  - Contextual appropriateness of answers
  - Completeness of information provided
  - Relevance to the original query

- **Operational Efficiency**:
  - Response time tracking in milliseconds
  - System resource utilization

### User-Centric Metrics

- **Satisfaction Tracking**:
  - 5-point rating system for user feedback
  - Qualitative comment collection
  - Trend analysis over time

### Continuous Improvement Process

- Built-in feedback collection mechanism
- Performance data aggregation and analysis
- KPI reporting with actionable insights

## 🔬 Technical Implementation

### Retrieval-Augmented Generation (RAG) Architecture

This chatbot utilizes a RAG architecture to provide accurate answers:

1. **Document Processing**: PDF documents are loaded and split into manageable chunks
2. **Vector Embeddings**: Text chunks are converted to vector embeddings using SentenceTransformer
3. **Semantic Search**: User questions are matched with the most relevant document sections
4. **Contextual Response Generation**: Retrieved context is used by the LLM to generate accurate, contextually-relevant answers

### Testing & Validation

The implementation includes a robust testing framework:

- **Automated Testing**: Script-based testing for core functionality
- **Human Evaluation**: User interface for collecting human judgments
- **Performance Benchmarking**: Tools for measuring and optimizing response times

### Assumptions Management

- **Data Handling**: Safe assumptions about document structure and availability
- **User Interaction**: Interface design catering to diverse user technical backgrounds
- **Scalability**: Considerations for expanding to multiple documents

## 🤝 Contributing

Contributions to improve the chatbot are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Aakash** - *Initial work* - [aakash4305](https://github.com/aakash4305)

## 🙏 Acknowledgments

- Research publication authors: Slamon DJ, Clark GM, Wong SG, Levin WJ, Ullrich A, McGuire WL
- SentenceTransformer and Milvus teams for their excellent tools
- Open-source LLM community

## 📞 Contact

For questions or feedback, please open an issue on this repository or contact the repository owner.
