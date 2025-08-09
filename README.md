
# 🎯 Face Recognition Attendance System

A **Python + FastAPI + Supabase** powered attendance system with **real-time face recognition**.  
Supports **roll number display**, **entry/exit tracking**, CSV exports, and a clean **Admin Dashboard**.

Developed by **Jeevan Gowda** 🚀

---

## 📸 Features

- **Face Recognition** using `face_recognition` & OpenCV
- **Separate Admin Dashboard** (`/`) and **Scanner UI** (`/scan`)
- **Roll Number Display** (not just UUIDs)
- **Entry & Exit Times** automatically updated
- **Supabase Integration** (students table + attendance_daily table)
- **CSV Export** of daily attendance
- **Enroll by UUID or Roll Number**
- **IST Timezone** handling

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **Database:** Supabase (PostgreSQL)
- **Face Recognition:** [face_recognition](https://github.com/ageitgey/face_recognition), OpenCV
- **Frontend:** HTML, TailwindCSS (for `/scan`), Vanilla JS

---

## 📂 Project Structure

```

.
├── app.py               # Main FastAPI app
├── admin.html           # Admin Dashboard UI
├── known\_faces/         # Stored known face images (uuid.jpg or roll\_no.jpg)
├── .env                 # Supabase credentials (not committed to git)
└── README.md            # Project documentation

````

---

## ⚙️ Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/yourusername/face-attendance.git
cd face-attendance
````

2️⃣ **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

4️⃣ **Set up `.env` file**

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
```

5️⃣ **Run the app**

```bash
uvicorn app:app --host 0.0.0.0 --port 9999 --reload
```

---

## 🖥️ Usage

* **Admin Dashboard**:
  Open [http://localhost:9999/](http://localhost:9999/) to view daily attendance & download CSV.

* **Face Scanner**:
  Open [http://localhost:9999/scan](http://localhost:9999/scan) to scan and recognize faces.

* **Enroll a Face by Roll Number**:

```bash
curl -X POST "http://localhost:9999/enroll_by_roll?roll_no=10A-001" \
     -F "file=@face.jpg"
```

* **Enroll a Face by UUID**:

```bash
curl -X POST "http://localhost:9999/enroll?student_id=<uuid>" \
     -F "file=@face.jpg"
```

---

## 📊 Supabase Table Structure

### students

| Column     | Type | Notes               |
| ---------- | ---- | ------------------- |
| id         | uuid | Primary key         |
| roll\_no   | text | Unique roll number  |
| full\_name | text | Student's full name |

### attendance\_daily

| Column      | Type   | Notes                 |
| ----------- | ------ | --------------------- |
| id          | bigint | Primary key           |
| student\_id | uuid   | FK → students.id      |
| att\_date   | date   | Attendance date (IST) |
| entry\_time | time   | First scan of the day |
| exit\_time  | time   | Last scan of the day  |
| notes       | text   | Optional remarks      |

---

## 📦 Requirements

* Python 3.10 or newer
* `face_recognition` (requires `dlib` installed)
* OpenCV
* FastAPI
* Supabase Python client

Example `requirements.txt`:

```txt
fastapi
uvicorn
supabase
python-dotenv
opencv-python
face_recognition
numpy
```

---

## 👨‍💻 Developer

**Jeevan Gowda**
💼 GitHub: [@jjgowda](https://github.com//jjgowda)
📧 Email: [cyberxstudios@protonmail.com](mailto:cyberxstudios@protonmail.com)

---

## 📜 License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute with attribution.


