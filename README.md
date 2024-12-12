# Backend Setup Guide

This guide provides step-by-step instructions to set up and run the backend for your Sentiment & Graph Visualizer project.

---

## **Prerequisites**

Before proceeding, ensure that you have the following installed on your system:

- **Python 3.10**
- **Pip** (Python package manager)

---

## **Setup Instructions**

Follow these steps to set up and run the backend:

### 1. **Create a Virtual Environment**
A virtual environment helps to manage project-specific dependencies.

```bash
python -m venv .venv
```

### 2. **Activate the Virtual Environment**
Activate the environment based on your operating system:

- **Linux/MacOS**:
  ```bash
  source .venv/bin/activate
  ```

- **Windows**:
  ```bash
  .venv\Scripts\activate
  ```

### 3. **Install Dependencies**
Install all required Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. **Download the spaCy Model**
Download the `en_core_web_sm` spaCy model, which is required for text preprocessing:

```bash
python -m spacy download en_core_web_sm
```

### 5. **Run the Backend Server**
Start the FastAPI backend using the following command:

```bash
uvicorn app:app --reload
```

- The server will run on **http://127.0.0.1:8000** by default.

---

## **Project Directory Structure**

Below is the suggested directory structure for your backend:

```
.
├── app.py                # Main backend application file
├── requirements.txt      # List of dependencies
├── .venv/                # Virtual environment (generated locally)
└── README.md             # Setup guide (this file)
```

---

## **Common Issues**

### 1. **spaCy Model Not Found**
- If you see the error `Can't find model 'en_core_web_sm'`, ensure you have downloaded the model by running:
  ```bash
  python -m spacy download en_core_web_sm
  ```

### 2. **Module Not Found Errors**
- Ensure you have activated the virtual environment and installed dependencies:
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## **Next Steps**

Once the backend is running, proceed to set up the frontend and connect it to this backend. For any issues, refer to the **Common Issues** section or contact the project maintainer.

---

### **Enjoy building your Sentiment & Graph Visualizer! 🚀**

