# 🎓 KI-System für Studierendenfragen (SPO Agent)



这是一个基于 RAG (检索增强生成) 的智能问答系统，用于回答关于 **通用考试条例 (SPO Allgemein)** 和 **机械制造/机电 (Maschinenbau)** 课程的问题。

项目技术栈：Streamlit + LangChain + OpenAI + FAISS。

------



## 🚀 快速开始 (Quick Start)



为了避免环境报错，请严格按照以下步骤配置环境。



### 1. 准备工作



确保你的电脑上安装了 **Python 3.9 或更高版本**。 推荐使用 [Anaconda](https://www.anaconda.com/) 或 Miniconda 来管理环境。



### 2. 获取代码



将项目文件夹下载到本地，并进入该目录：

Bash

```
cd Ki_agent
```



### 3. 环境配置 (Windows & Mac 通用推荐)



我们强烈建议创建一个新的虚拟环境，不要使用系统默认 Python 环境。



#### 方案 A：使用 Conda (推荐)



打开终端 (Terminal) 或 Anaconda Prompt，执行以下命令：

Bash

```
# 1. 创建名为 Ki_agent 的环境，指定 python 3.10
conda create -n Ki_agent python=3.10

# 2. 激活环境
conda activate Ki_agent

# 3. 安装所有依赖库 (一定要在项目根目录下运行)
pip install -r requirements.txt
```



#### 方案 B：使用原生 Python venv



如果你没有安装 Conda，请使用以下命令：

**Windows:**

Bash

```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux:**

Bash

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

------



### 4. 配置 API Key 🔑



本项目需要 OpenAI API Key 才能运行。

1. 在项目根目录下创建一个名为 `.env` 的文件（注意前面有个点）。
2. 用记事本或代码编辑器打开它，输入你的 Key：

代码段

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

*(请向项目负责人索要 Key，或使用自己的 Key)*

------



### 5. 运行项目 ▶️



确保你的终端显示环境已激活（例如左侧有 `(Ki_agent)` 字样），然后运行：

Bash

```
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，你就可以开始对话了！

------



## 📂 项目结构



Plaintext

```
Ki_agent/
├── data/                  # 存放 PDF 源文件 (SPO文档)
│   ├── AllgemeinerTeil...pdf
│   └── Maschinenbau...pdf
├── app.py                 # 主程序代码
├── .env                   # 配置文件 (不要上传到 GitHub!)
├── requirements.txt       # 依赖库列表
└── README.md              # 说明文档
```

------



## ❓ 常见问题 (Troubleshooting)



**Q1: 报错 `ModuleNotFoundError: No module named 'langchain_community'`**

- **原因**：依赖库没装全。
- **解决**：请确认你激活了虚拟环境，并重新运行 `pip install -r requirements.txt`。

**Q2: 报错 `Could not import faiss python package`**

- **原因**：缺少向量数据库工具。
- **解决**：运行 `pip install faiss-cpu`。

**Q3: 报错 `pypdf package not found`**

- **原因**：缺少 PDF 读取工具。
- **解决**：运行 `pip install pypdf`。

**Q4: 运行后立刻报错，显示 OpenAI 相关错误**

- **原因**：`.env` 文件没配置好，或者 Key 余额不足。
- **解决**：检查 `.env` 文件名是否正确（必须是 `.env`），以及 Key 是否有效。