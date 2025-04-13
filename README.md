# 🛰️ Spectra

Spectra is a next-generation computer vision platform that transforms passive surveillance into **proactive intelligence**. Designed for modern public spaces like airports, stadiums, and transit hubs, Spectra detects high-risk situations as they unfold—so operators can focus on what matters most.

## 🚨 Why Spectra?

Traditional surveillance relies heavily on human attention: dozens of screens, hours of footage, and constant vigilance. Spectra upgrades this with real-time detection and context-aware intelligence.

### Spectra can detect:

- 🧍‍♂️ A person collapsing in a crowd  
- 🏃 Panic-induced running or sudden movements  
- 🎒 Unattended or hazardous objects  
- 👥 Dense crowd formations signaling unrest  
- 🔍 Queries like “person in red jacket near Gate 3”  
- 📝 Overlay dynamic labels or alerts on live feeds  
- 📢 Push real-time notifications with actionable insights  

Spectra turns **cameras into collaborators**, enabling smarter, faster responses.

---

## 🛠️ How It Works

### 🔹 Frontend  
- **React** for a dynamic and responsive UI  
- Connected directly to **AWS S3** via the AWS SDK for file storage and retrieval  

### 🔹 Backend  
- **Roboflow** for object and scene analysis  
- **Groq API (LLaMA 4 Scout)** for frame-level summarization and classification  
- Audio processing with **pydub**  
- Custom **semantic vector search** using **CLIP-style embeddings**  
- Text-video search using **FAISS** with L2 norm distance metrics  

### 🔹 Database  
- Cloud-based storage via **Amazon S3**  
- Semantic video indexing with a Flask API  
- Embeddings aligned for cross-modal (text-image) retrieval  

---

## 🚧 Challenges

- ❗ Originally planned a custom database but pivoted to AWS S3 for simplicity and scale  
- 🔌 Faced integration issues when porting vector search database from RCAC to Modal—resolved using Flask  
- 🔁 Worked through async challenges and optimized API calls  

---

## 🌟 What We’re Proud Of

- Built full-stack video search + alert platform from scratch  
- Created a scalable pipeline using **React + AWS + vision-language models**  
- Developed a custom vector search engine with minimal external libraries  
- Integrated multiple APIs (Groq, RoboFlow) into a seamless experience  

---

## 🧠 What We Learned

- **React + AWS SDK** integration and frontend S3 uploads  
- **Semantic search** using CLIP and FAISS  
- Real-time **pose and motion detection**  
- **Async programming**, cloud APIs, and GPU computing with Slurm and CUDA  

---

## 🚀 What's Next?

- **Live CCTV feed integration**  
- **Expanded detection**: smoke, weapons, vehicles, emotions  
- **Custom rule engines** for authorities  
- **Multi-camera tracking** for person/path re-identification  
- **Privacy-preserving AI** (edge processing, anonymization)  
- **Mobile dashboards** for first responders  
- **Multilingual support** for prompt-based search  
- **Plugin architecture** for 3rd-party integration  

---

## 🔧 Built With

[![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![Groq](https://img.shields.io/badge/Groq-FF6F00?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Whisper](https://img.shields.io/badge/Whisper-0041C2?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Roboflow](https://img.shields.io/badge/Roboflow-101010?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB...)](https://roboflow.com/)
[![Pydub](https://img.shields.io/badge/Pydub-FFDD57?style=for-the-badge&logo=python&logoColor=black)](https://github.com/jiaaro/pydub)

