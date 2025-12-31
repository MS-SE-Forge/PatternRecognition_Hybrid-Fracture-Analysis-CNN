# Hybrid Fracture Analysis System 🦴

Welcome! This is a smart tool that uses **Artificial Intelligence (AI)** to look at X-ray images and help doctors decide:
1.  **Is the bone broken?** (Fracture Detection)
2.  **How bad is it?** (Severity Analysis)

It combines **Deep Learning** (like an intuitive brain) with **Mathematics** (like a precise ruler) to give the best possible diagnosis.

---

## 📚 Table of Contents
1.  [How it Works (Simply Explained)](#-how-it-works-simply-explained)
2.  [Installation (Getting Started)](#%EF%B8%8F-installation--setup-for-beginners)
3.  [Training (Teaching the Brain)](#-how-to-train-the-brain-step-1)
4.  [Usage (Analyzing X-rays)](#-how-to-use-it-step-2)
5.  [Troubleshooting](#-troubleshooting)

---

## 🧠 How it Works (Simply Explained)
Imagine two doctors looking at an X-ray together:

1.  **Dr. AI (The Pattern Spotter)**:
    *   This is a computer program (CNN) that has studied thousands of X-ray images.
    *   It looks at the overall picture and instantly says: *"I think I see a break here!"* or *"This looks healthy."*

2.  **Dr. Math (The Measurer)**:
    *   This part acts like a ruler. It looks specifically at the white lines of the bone.
    *   It measures the gap between broken pieces in millimeters.
    *   **Big Gap (>2mm)** = Severe (Needs Surgery).
    *   **Tiny Gap (<1mm)** = Hairline (Needs a Cast).

 **The Judge**:
 *   We combine both opinions. If Dr. AI sees a break AND Dr. Math finds a gap, we know exactly how serious it is.

---

## 🛠️ Installation & Setup (For Beginners)

### 1. Install Python
You need Python (the programming language) installed.
*   **Check if you have it**: Open your computer's terminal (Command Prompt) and type `python --version`.
*   **If not**: Download it from [python.org](https://www.python.org/downloads/).

### 2. Get the Code
You can download this project in two ways:
*   **Option A (Easy)**: Download the ZIP file and unzip it.
*   **Option B (Developer)**: Clone it using Git:
    ```bash
    git clone https://github.com/MS-SE-Forge/PatternRecognition_Hybrid-Fracture-Analysis-CNN.git
    ```

### 3. Install the "Brain" Libraries
Open your terminal inside the project folder and run this magic command. It installs all the tools the AI needs to see and think:

```bash
pip3 install torch torchvision opencv-python numpy scikit-image pillow
```

*(Note: If you see "command not found", try `pip3` instead of `pip`)*

---

## 🏫 How to Train the Brain (Step 1)
The AI is currently "blank". You need to teach it using your collection of X-rays.

### A. Organize Your Data
Go to the `data` folder inside this project. Make sure your folders look **EXACTLY** like this tree. Spelling matters!

```text
Project/
├── src/               <-- Code stays here
├── data/
│   ├── train/         <-- 80% of your images (Study Material)
│   │   ├── fractured/
│   │   └── normal/
│   ├── val/           <-- 10% of your images (Practice Quiz)
│   │   ├── fractured/
│   │   └── normal/
│   └── test/          <-- 10% of your images (Final Exam)
│       ├── fractured/
│       └── normal/
```

### B. Start Training
Type this into your terminal:

```bash
python src/train.py
```

**What will happen?**
1.  **Epochs**: You will see a counter go from 1 to 10 (or more). An "Epoch" is one full round of studying all the images.
2.  **Accuracy (Acc)**: You want this number to go UP (e.g., `0.50` -> `0.85` -> `0.92`). `0.92` means it is 92% correct!
3.  **Loss**: You want this number to go DOWN.
4.  **Final Result**: When it finishes, it will say `Test Set Accuracy: ...`. This is the final grade.
5.  **The File**: It creates a file named `fracture_model_best.pth`. **Do not delete this.** This file contains the AI's learned intelligence.

---

## 🚀 How to Use It (Step 2)
Now that the brain is trained, you can give it NEW images to analyze.

### Method 1: Batch Analysis (Recommended)
This is for processing many images at once (e.g., a whole folder of new patients).

1.  Create a folder named `data/inference_input` (or clean it if it exists).
2.  Put all your new X-ray images (`.jpg`, `.png`) inside it.
3.  Run the main command:

    ```bash
    python src/main.py
    ```

4.  **See Results**: Go to the folder `data/inference_results`. You will find a text file for each image (e.g., `patient_X_result.txt`) telling you if they have a fracture.

---

## ❓ Troubleshooting

| Problem | Solution |
| :--- | :--- |
| **"No module named..."** | You missed the installation step. Run `pip install ...` again. |
| **"Dataset directory not found"** | You didn't separate your images correctly. Check that you have `data/train` and NOT just `train` sitting alone. |
| **"RuntimeError" (during training)** | Your images might be corrupted or in a weird format. Ensure they are standard `.jpg` or `.png`. |
| **Accuracy is very low (<0.6)** | You need more data! Or your "Fractured" and "Normal" folders might be mixed up. |
| **System is slow** | This is normal if you don't have a GPU (Graphics Card). AI needs a lot of math power. |

---

## 💻 Hardware Recommendations

Can you run this on your laptop? **Yes!** But speed varies:

| Hardware | Speed | Notes |
| :--- | :--- | :--- |
| **Mac (M1/M2/M3)** | ⭐⭐⭐ | Good balance. The code now supports Apple Silicon naturally! |
| **NVIDIA GPU** | ⭐⭐⭐⭐⭐ | The gold standard. Training will be lightning fast. |
| **Standard Laptop (CPU)** | ⭐ | It works, but might take 10-20 mins to train instead of 1 min. |

**Tip**: If your computer is too slow, you can use [Google Colab](https://colab.research.google.com/) for free GPU access.

### 🌐 How to use Google Colab
1.  Go to [colab.research.google.com](https://colab.research.google.com/).
2.  Click **File > Upload Notebook**.
3.  Upload the file `notebooks/fracture_analysis_colab.ipynb` from this project.
4.  Run the cells! (The data will be downloaded automatically from GitHub).

---

## ☁️ Deployment (Advanced)
*   **Docker**: Use the provided `Dockerfile` to wrap this up for easy shipping.
*   **Cloud API**: The code can be wrapped in `FastAPI` to serve requests over the internet.

Enjoy your AI Assistant! 🩺
