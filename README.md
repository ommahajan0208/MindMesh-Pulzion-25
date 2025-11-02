# MindMesh — YouTube Trend Analyzer (Pulzion'25)

## Overview
**MindMesh** is an AI-powered dashboard that helps creators, marketers, and data enthusiasts analyze **YouTube trending videos** for any country.  
It provides real-time insights on trending categories, popular keywords, engagement analytics, and even generates **AI-driven content ideas** using **Gemini 2.5 Flash**.

> Built for **Pulzion'25 Hackathon** 

---

## Core Features

### Frontend (Data Dashboard)
- **Fetch Trending Videos:** Real-time data from the YouTube Data API (country-specific).
- **Video Dashboard:** Displays video thumbnails, titles, and engagement metrics.
- **Keyword Analysis:** Extracts trending keywords from video titles.
- **Category Visualization:** Shows category-wise video counts using interactive charts.

### Backend (AI Insights)
- **Creator Coach AI:** Uses Gemini 2.5 Flash to analyze YouTube trends and give personalized growth advice.
- **Content Suggestion Engine:** Clusters trending videos to generate **new video ideas** with title, hook, and thumbnail suggestions.
- **Upload Recommendations:** Finds best times and categories to upload for higher engagement.

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| **Backend** | Flask (Python) |
| **API** | YouTube Data API v3 |
| **AI Model** | Gemini 2.5 Flash (Google DeepMind) |
| **Data Analysis** | Pandas, NumPy, NLTK, Scikit-learn, TextBlob |
| **Visualization (Frontend)** | Chart.js / Recharts (via React or any frontend client) |
| **Environment Management** | python-dotenv |

---

## ⚙️ Setup Instructions

### 1️. Clone the Repository
```bash
git clone https://github.com/your-username/MindMesh-Pulzion25.git
cd MindMesh-Pulzion25
```

### 2️. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3️. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️. Add API Keys
Create a `.env` file in the root directory and add:

```
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 5️. Run the Flask Server
```bash
python app.py
```
Server will run at: **http://127.0.0.1:5000/**

---
## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/get_trending_data?country=US` | Fetches and analyzes trending videos for a given country. |
| `/get_creator_suggestions?country=US` | Generates AI-powered video ideas from trending patterns. |
| `/get_creator_coach?country=US&genre=10` | Returns Creator Coach insights (Gemini-based). |

---

## My Contributions

As part of **Team MindMesh** at Pulzion'25, I was responsible for:

 **Backend Development:** Built the Flask-based API endpoints (`/get_trending_data`, `/get_creator_coach`) to fetch and process YouTube data.

 **AI Integration:** Integrated the Gemini 2.5 Flash model for generating content suggestions and creator insights.

 **Data Analysis:** Implemented keyword extraction, engagement metrics, and trend clustering using Python libraries (NLTK, TextBlob, scikit-learn).

 **Testing & Optimization:** Improved API response times and ensured reliable JSON outputs for frontend integration.

---

## Team MindMesh

Built with ❤️ by innovative minds at **Pulzion'25 Hackathon**. Focused on merging data analytics + AI creativity for YouTube creators.

