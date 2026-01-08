# mOrpheus Virtual Assistant

mOrpheus is a powerful, flexible, and real-time voice assistant designed to run locally or via cloud APIs. It bridges the gap between local privacy-focused AI and cutting-edge cloud models, offering a seamless voice interaction experience.

## Features

-   **Dual Operation Modes**:
    -   **Local Mode**: Runs completely offline using `OpenAI Whisper` for speech-to-text and `LM Studio` for the LLM and TTS.
    -   **Realtime Mode**: leverages OpenAI's or Azure's Realtime API for low-latency, high-quality interactions.
-   **Wake Word Detection**: Built-in hotword detection (default: "Hey Cassie") for hands-free operation.
-   **Voice Activity Detection (VAD)**: Smartly detects when you stop speaking to process your query automatically.
-   **Flexible Interaction**: Supports Push-to-Talk, Hotword, or both simultaneously.
-   **High-Quality TTS**: Integrated support for expressive Text-to-Speech models.

## Prerequisites

-   **Python 3.10+**
-   **CUDA-capable GPU** (highly recommended for Local Mode)
-   **LM Studio** (for Local Mode) running a compatible LLM and TTS server.
-   **API Keys** (for Realtime Mode) if using OpenAI or Azure services.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Real-Time-Local-Voice-AI.git
    cd Real-Time-Local-Voice-AI
    ```

2.  **Install PyTorch**:
    Depending on your CUDA version, install the appropriate PyTorch version. For example, for CUDA 12.6:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Configuration is managed via `settings.yml`. You can customize:

-   **Pipeline Mode**: `local` or `realtime`.
-   **Whisper Settings**: Model size (`small.en`, `large-v3`, etc.) and sample rate.
-   **LM Studio Settings**: API URL, model identifiers, and generation parameters.
-   **Realtime API**: Endpoints and API keys for OpenAI/Azure.
-   **Audio Settings**: Input/output device selection and hotword sensitivity.
-   **VAD Settings**: Adjust silence thresholds and aggressiveness.

### Example `settings.yml` (snippet)
```yaml
pipeline:
  mode: "local"

whisper:
  model: "small.en"

lm:
  api_url: "http://127.0.0.1:1234/v1"
  chat:
    model: "gemma-3-12b-it"
```

## Usage

Start the assistant by running the main script:

```bash
python main.py
```

To specify a custom configuration file:

```bash
python main.py --config my_settings.yml
```

## License

[MIT License](LICENSE)