# ⚡ Partial Discharge Classification with Deep Learning

> **Deep Learning system for automatic classification of partial discharge (PD) patterns in high-voltage equipment**

This project applies **Convolutional Neural Networks (CNNs)** to classify **phase-resolved partial discharge (PRPD) patterns** used in high-voltage diagnostics.  
It was originally developed as part of my **Master’s thesis at the University of Stuttgart** and is shared here as a **technical portfolio project**.

---

## 🚀 Why this project matters

Partial discharges are an early indicator of insulation defects in electrical equipment.  
In practice, PRPD patterns are still **interpreted manually by experts**, which is slow and hard to scale.

This project shows how **deep learning can automate PD pattern recognition**, enabling:

- Continuous condition monitoring  
- Faster fault detection  
- Reduced dependency on expert interpretation  

---

## 🧠 What this project does

- Classifies PRPD patterns represented as images
- Supports **multi-label classification** (multiple fault types in one pattern)
- Includes a **desktop GUI** for training, testing, and visualization
- Benchmarks multiple CNN architectures
- Designed with **real-time usage** in mind

---

## 📊 Key Results

- ✅ **100% accuracy** for images containing a **single PD pattern**
- ⚡ For images with **two superimposed PD patterns**:
  - ~30%: both fault types correctly detected
  - ~70%: at least one fault type correctly detected
- Demonstrates strong suitability for **online monitoring systems**

---

## 🖥️ GUI Preview

The project includes a PyQt-based desktop application that allows users to:

- Configure datasets and hyperparameters  
- Train CNN models  
- Test single images or entire folders  
- Visualize PRPD patterns and predictions  

*(Add screenshots here)*

---

## 🧩 Model & Approach

- Image-based classification of PRPD patterns
- Convolutional Neural Networks (CNNs)
- Multi-label output with optimized decision thresholds
- Evaluation using **Hamming Loss** and per-class metrics
- Synthetic generation of multi-fault images by overlaying patterns

---

## 🛠️ Tech Stack

- Python  
- Keras (TensorFlow backend)  
- Convolutional Neural Networks  
- PyQt  
- NumPy, Matplotlib, scikit-learn  

> ⚠️ Note  
> This project was developed in **2016–2017** using TensorFlow 1.x / Keras 2.x and is shared for educational and portfolio purposes.

---

## 🕰️ Project Background

- 🎓 Master’s Thesis  
- 🏫 University of Stuttgart  
- 🔌 Institute for Power Transmission and High Voltage Technology  
- 📅 2016 – 2017  

---

## 🔮 How I would build this today

If rebuilt today, I would:

- Migrate to **TensorFlow 2 or PyTorch**
- Use **EfficientNet or Vision Transformers**
- Add **Grad-CAM explainability**
- Track experiments with **Weights & Biases**
- Provide a **REST API (FastAPI)** for inference
- Package everything using **Docker**

---

## 👤 About me

**Wilhelm Ziegler**  
Software Engineer with a strong background in:

- Deep Learning & AI  
- Automation & system design  
- Applied engineering software  

📍 Stuttgart, Germany  
🔗 LinkedIn: *https://www.linkedin.com/in/wilhelm-ziegler/*
📧 Email: *w_ziegler@outlook.com*  

---

## 📄 License

MIT License