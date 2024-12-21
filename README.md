# Text Summarization with Google Pegasus

This repository demonstrates a **Text Summarization Pipeline** using the [Google Pegasus](https://huggingface.co/google/pegasus-cnn_dailymail) model from the `transformers` library. The pipeline leverages state-of-the-art Natural Language Processing (NLP) and Deep Learning techniques to summarize large text datasets effectively.

## Features

- **Model**: Fine-tunes `google/pegasus-cnn_dailymail` for abstractive summarization tasks.
- **Pipeline Components**:
  - Data Ingestion
  - Data Transformation
  - Model Training
  - Evaluation
- **Tools Used**:
  - [Hugging Face Transformers](https://huggingface.co/transformers)
  - [Datasets Library](https://huggingface.co/docs/datasets/)
  - [PyTorch](https://pytorch.org/)
  - [Accelerate](https://github.com/huggingface/accelerate)
  - [Tokenizers](https://github.com/huggingface/tokenizers)
  - [Yaml](https://pyyaml.org/)
  - Python Multiprocessing

## Achievements

- Successfully fine-tuned the `google/pegasus-cnn_dailymail` model for abstractive text summarization on the SAMSUM dialogue dataset.
- Achieved competitive results in abstractive summarization with evaluation metrics such as Rouge-1, Rouge-2, and Rouge-L.
- Designed and implemented an NLP pipeline capable of handling large-scale dialogue datasets effectively.
- Integrated advanced tokenization strategies to preprocess complex dialogue structures, enhancing model efficiency and performance.
- Leveraged transfer learning with pre-trained Pegasus models, significantly reducing the computational resources required for fine-tuning.
- Applied advanced deep learning techniques, such as gradient clipping and learning rate scheduling, to stabilize training and improve convergence.
- Used distributed training techniques with the `Accelerate` library, enabling efficient utilization of multi-GPU setups.
- Conducted in-depth evaluation using automated metrics like Rouge and manual review to validate the quality of generated summaries.
- Generated insightful visualizations of training metrics and evaluation results to communicate the model's performance effectively.
- Developed reusable modular components for ingestion, transformation, and evaluation, promoting scalability and extensibility for future NLP tasks.


## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KARSarma/textsummarizer.git
   cd textsummarizer
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the pipeline:
```bash
python main.py
```

### Update configurations:
- Modify `config/config.yaml` for paths.
- Adjust `params.yaml` for training hyperparameters.

### Model outputs:
- Saved in the `artifacts/model_trainer` directory.
- Logs and summarized outputs are available for review.

## Libraries and Technologies

- **Hugging Face Transformers**: Pre-trained models for text summarization.
- **PyTorch**: Flexible deep learning framework.
- **Datasets**: Efficient handling of large datasets.
- **Accelerate**: Distributed and efficient training.
- **Tokenizers**: Fast text preprocessing.
- **Yaml**: Configuration management.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pre-trained models and tools.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- Community contributions for datasets and improvements.

This project showcases how state-of-the-art NLP models can be utilized for text summarization tasks efficiently.


