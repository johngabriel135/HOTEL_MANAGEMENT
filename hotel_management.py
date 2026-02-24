import tkinter as tk
from tkinter import messagebox, filedialog
from turtle import title
from PIL import Image, ImageTk
from time import strftime
import sqlite3
import tkinter.ttk as ttk
import datetime
import os
import random
import shutil
import webbrowser
import uuid
import hashlib
from altair import value
import cv2
import numpy as np
import pickle
import json
import requests
import threading

# ===== FACE RECOGNITION SETUP =====
try:
    import cv2
    import numpy as np
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: OpenCV not installed. Face recognition disabled. Install with: pip install opencv-python")

# Face recognition database
FACE_DB_DIR = "face_database"
FACE_ENCODINGS_FILE = os.path.join(FACE_DB_DIR, "face_encodings.pkl")

def ensure_face_db_dir():
    """Create face database directory if it doesn't exist"""
    if not os.path.exists(FACE_DB_DIR):
        os.makedirs(FACE_DB_DIR, exist_ok=True)

def capture_face_encoding(username):
    """Capture multiple face frames for a user and average them for better accuracy"""
    if not FACE_RECOGNITION_AVAILABLE:
        return False
    
    ensure_face_db_dir()
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            show_error_message("Cannot access webcam. Face registration skipped.")
            return False
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print(f"Capturing face for {username}. Hold still for 5 seconds...")
        
        start_time = datetime.datetime.now()
        face_encodings = []  # Store multiple encodings
        face_detected = False
        
        # Create persistent camera window
        cv2.namedWindow(f'Face Registration - {username}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Face Registration - {username}', 640, 480)
        
        while (datetime.datetime.now() - start_time).total_seconds() < 5:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Add timer text to frame
            elapsed = int((datetime.datetime.now() - start_time).total_seconds())
            cv2.putText(frame, f"Time: {elapsed}s / 5s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if len(faces) > 0:
                face_detected = True
                # Draw rectangle around face
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Compute face encoding (average pixel values in face region)
                face_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                face_encoding = np.mean(face_roi)  # Single scalar value
                face_encodings.append(face_encoding)
            else:
                cv2.putText(frame, "No face detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow(f'Face Registration - {username}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not face_detected or len(face_encodings) == 0:
            show_error_message("No face detected. Please try again with better lighting.")
            return False
        
        # Average all captured encodings for better accuracy
        avg_encoding = np.mean(face_encodings)
        encodings_file = os.path.join(FACE_DB_DIR, f"{username}_face.npy")
        np.save(encodings_file, avg_encoding)
        
        show_success_message(f"Face registered successfully! ({len(face_encodings)} frames captured)")
        print(f"Saved face encoding for {username}: {avg_encoding}")
        return True
    
    except Exception as e:
        show_error_message(f"Error capturing face: {str(e)}")
        return False

def recognize_face():
    """Attempt to recognize a face from webcam"""
    if not FACE_RECOGNITION_AVAILABLE:
        return None
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Timeout: 5 seconds to capture face
        start_time = datetime.datetime.now()
        while (datetime.datetime.now() - start_time).total_seconds() < 5:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Process first detected face
                (x, y, w, h) = faces[0]
                face_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                current_encoding = np.mean(face_roi, axis=(0, 1))
                
                # Compare with stored encodings
                best_match = None
                best_distance = float('inf')
                
                for user_file in os.listdir(FACE_DB_DIR):
                    if user_file.endswith('_face.npy'):
                        username = user_file.replace('_face.npy', '')
                        try:
                            stored_encoding = np.load(os.path.join(FACE_DB_DIR, user_file))
                            distance = np.linalg.norm(current_encoding - stored_encoding)
                            
                            # If distance is below threshold (0.3), it's a match
                            if distance < 0.3 and distance < best_distance:
                                best_match = username
                                best_distance = distance
                        
                        except Exception:
                            pass
                
                cap.release()
                cv2.destroyAllWindows()
                return best_match
            
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    except Exception as e:
        print(f"Face recognition error: {e}")
        return None

# ===== END FACE RECOGNITION =====# ===== FACE RECOGNITION SETUP =====

# Color scheme: Modern dark theme with gold accents
COLOR_DARK_BG = "#1a1a1a"       # Very dark gray/black for main backgrounds
COLOR_SIDEBAR = "#112439"        # Darker blue for sidebar and buttons
COLOR_SIDEBAR_HOVER = "#3A3F88"  # Even darker blue for hover/highlight
COLOR_BLUE_PRIMARY = "#1e40af"   # Professional blue (alternate)
COLOR_BLUE_LIGHT = "#3b82f6"     # Lighter blue for hover
COLOR_GOLD = "#f59e0b"           # Gold for accents
COLOR_WHITE_CARD = "#ffffff"     # White for cards
COLOR_GRAY = "#6b7280"           # Gray for text
COLOR_TEXT_DARK = "#111827"      # Dark text on white cards

# try to use bcrypt if available, otherwise fall back to PBKDF2
try:
    import bcrypt
    _HAS_BCRYPT = True
except Exception:
    bcrypt = None
    _HAS_BCRYPT = False

# connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("hotel_management.db")
cursor = conn.cursor()
# session globals
current_user = None
current_user_role = None
open_windows = []
_kpi_refresh_job = None
kpi_labels = {}
_poll_job = None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt if available, otherwise PBKDF2.
    Returns a string representation suitable for storage.
    """
    if not password:
        return ""
    try:
        if _HAS_BCRYPT:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        # fallback to pbkdf2
        salt = os.urandom(16)
        dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
        return f"pbkdf2${salt.hex()}${dk.hex()}"
    except Exception:
        # as last resort, return hex of sha256 (non-ideal)
        return hashlib.sha256(password.encode('utf-8')).hexdigest()


def verify_password(password: str, stored: str) -> bool:
    """Verify cleartext password against stored hash (bcrypt or pbkdf2)."""
    if not stored or not password:
        return False
    try:
        if _HAS_BCRYPT and stored.startswith('$2'):
            return bcrypt.checkpw(password.encode('utf-8'), stored.encode('utf-8'))
        if stored.startswith('pbkdf2$'):
            try:
                _, salt_hex, dk_hex = stored.split('$', 2)
                salt = bytes.fromhex(salt_hex)
                expected = bytes.fromhex(dk_hex)
                dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
                return dk == expected
            except Exception:
                return False
        # last resort: compare sha256 hex or plaintext (legacy)
        if len(stored) == 64:
            return hashlib.sha256(password.encode('utf-8')).hexdigest() == stored
        return password == stored
    except Exception:
        return False

def start_polling(interval_ms=2000):
    global _poll_job
    def _tick():
        try:
            # refresh presence in DB
            now = datetime.datetime.utcnow().isoformat()
            if current_user:
                cursor.execute("UPDATE users SET last_active=? WHERE username=?", (now, current_user))
                conn.commit()
        except Exception:
            pass
        # chat functionality removed
        pass
        try:
            _poll_job = root.after(interval_ms, _tick)
        except Exception:
            pass
    _tick()

def stop_polling():
    global _poll_job
    try:
        if _poll_job:
            root.after_cancel(_poll_job)
            _poll_job = None
    except Exception:
        pass
# create the users table
cursor.execute ("""CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL
    )
    """)
conn.commit()

# ensure users table has last_active column for presence tracking
try:
    cursor.execute("ALTER TABLE users ADD COLUMN last_active TEXT")
    conn.commit()
except sqlite3.OperationalError:
    # column already exists
    pass

# messages table for in-app chat
cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender TEXT NOT NULL,
    receiver TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    is_read INTEGER DEFAULT 0
)
''')

# notifications table
cursor.execute('''CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_read INTEGER DEFAULT 0
)
''')
conn.commit()

# invoices table
cursor.execute('''CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY,
    booking_id TEXT,
    guest_name TEXT,
    amount REAL,
    status TEXT,
    created_at TEXT
)
''')

conn.commit()

# settings table for app preferences
cursor.execute('''CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
)
''')
# seed default settings
try:
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", ("auto_refresh_kpis", "1"))
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", ("notifications_enabled", "1"))
    conn.commit()
except Exception:
    pass

# staff's database setup
cursor.execute("SELECT * FROM users WHERE username = ?", ("Staff",))
if not cursor.fetchone():
    cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", ("Staff", "staff123", "staff"))

# admin's database setup
cursor.execute("select * from users WHERE username =?",("Admin",))
if not cursor.fetchone():
    cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",("Admin", "admin123", "Admin"))

# receptionist's database setup
cursor.execute("select * from users WHERE username =?",("Receptionist",))
if not cursor.fetchone():
    cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",("Receptionist", "customer", "Receptionist"))
    conn.commit()
# migrate any plaintext passwords to hashed format (best-effort)
try:
    cursor.execute("SELECT username, password FROM users")
    rows = cursor.fetchall()
    for u, pw in rows:
        if not pw:
            continue
        # if looks like bcrypt or pbkdf2, skip
        if (isinstance(pw, str) and (pw.startswith('$2') or pw.startswith('pbkdf2$') or len(pw) == 64)):
            continue
        # otherwise re-hash and update
        try:
            newhash = hash_password(pw)
            cursor.execute("UPDATE users SET password=? WHERE username=?", (newhash, u))
        except Exception:
            pass
    conn.commit()
except Exception:
    pass
    
# create the users rooms table
cursor.execute ("""CREATE TABLE IF NOT EXISTS user_rooms (
    room_number INTEGER NOT NULL,
    room_type TEXT NOT NULL,
    room_price INTEGER NOT NULL,
    room_status TEXT NOT NULL
    )
    """)

# Pre-fill the database with fixed rooms and default room numbering
default_rooms = [
    (101, "Single", 100, "Available"),
    (102, "Single", 100, "Available"),
    (103, "Single", 100, "Available"),
    (201, "Double", 150, "Available"),
    (202, "Double", 150, "Available"),
    (203, "Double", 150, "Available"),
    (301, "Suite", 250, "Available"),
    (302, "Suite", 250, "Available"),
    (303, "Suite", 250, "Available")
]
for room in default_rooms:
    cursor.execute("SELECT * FROM user_rooms WHERE room_number = ?", (room[0],))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO user_rooms (room_number, room_type, room_price, room_status) VALUES (?, ?, ?, ?)", (room[0], room[1], room[2], room[3]))
    else:
        cursor.execute("""UPDATE user_rooms SET room_type = ?, room_price = ?, room_status = ? WHERE room_number = ?""",
                       (room[1], room[2], room[3], room[0]))
        conn.commit()
        
# create the guests table (for guest profiles)
cursor.execute("""CREATE TABLE IF NOT EXISTS guests (
    guest_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_name TEXT NOT NULL UNIQUE,
    phone TEXT,
    email TEXT,
    address TEXT,
    created_at TEXT NOT NULL
    )
    """)
conn.commit()

# create the guest bookings table
cursor.execute ("""CREATE TABLE IF NOT EXISTS guest_bookings (
    booking_id TEXT PRIMARY KEY,
    guest_name TEXT NOT NULL,
    room_number INTEGER NOT NULL,
    check_in_date TEXT NOT NULL,
    check_out_date TEXT NOT NULL,
    status TEXT NOT NULL,
    total_amount INTEGER NOT NULL,
    FOREIGN KEY (room_number) REFERENCES user_rooms (room_number)
    )
    """)
conn.commit()

# Ensure guest_bookings.booking_id is TEXT (migrate if older schema used INTEGER PK)
def migrate_guest_bookings_schema():
    try:
        cur = conn.cursor()
        info = list(cur.execute("PRAGMA table_info('guest_bookings')"))
        if info:
            # info rows: (cid, name, type, notnull, dflt_value, pk)
            booking_col = next((r for r in info if r[1] == 'booking_id'), None)
            if booking_col and booking_col[2].upper() != 'TEXT':
                # perform migration: create new table with TEXT booking_id and copy data
                cur.execute('''CREATE TABLE IF NOT EXISTS guest_bookings_new (
                    booking_id TEXT PRIMARY KEY,
                    guest_name TEXT NOT NULL,
                    room_number INTEGER NOT NULL,
                    check_in_date TEXT NOT NULL,
                    check_out_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_amount INTEGER NOT NULL,
                    FOREIGN KEY (room_number) REFERENCES user_rooms (room_number)
                )''')
                # copy rows, casting booking_id to TEXT
                cur.execute("INSERT OR REPLACE INTO guest_bookings_new (booking_id, guest_name, room_number, check_in_date, check_out_date, status, total_amount) SELECT CAST(booking_id AS TEXT), guest_name, room_number, check_in_date, check_out_date, status, total_amount FROM guest_bookings")
                cur.execute("DROP TABLE guest_bookings")
                cur.execute("ALTER TABLE guest_bookings_new RENAME TO guest_bookings")
                conn.commit()
    except Exception:
        pass

migrate_guest_bookings_schema()

# create the restaurant_orders table
cursor.execute("""CREATE TABLE IF NOT EXISTS restaurant_orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_number TEXT NOT NULL,
    items TEXT NOT NULL,
    status TEXT NOT NULL,
    total_price REAL NOT NULL,
    order_date TEXT NOT NULL
    )
    """)

# create the reviews table
cursor.execute("""CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id TEXT NOT NULL,
    guest_name TEXT NOT NULL,
    rating INTEGER NOT NULL,
    feedback TEXT NOT NULL,
    review_date TEXT NOT NULL,
    FOREIGN KEY (booking_id) REFERENCES guest_bookings (booking_id)
    )
    """)

# Create housekeeping tasks table
cursor.execute("""CREATE TABLE IF NOT EXISTS housekeeping_tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_number INTEGER NOT NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    notes TEXT,
    FOREIGN KEY (room_number) REFERENCES user_rooms (room_number)
    )
    """)

# Create maintenance issues table
cursor.execute("""CREATE TABLE IF NOT EXISTS maintenance_issues (
    issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_number INTEGER,
    issue_type TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    reported_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (room_number) REFERENCES user_rooms (room_number)
    )
    """)

# Create customer feedback table
cursor.execute("""CREATE TABLE IF NOT EXISTS customer_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id TEXT NOT NULL,
    guest_name TEXT NOT NULL,
    feedback_type TEXT NOT NULL,
    message TEXT NOT NULL,
    rating INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (booking_id) REFERENCES guest_bookings (booking_id)
    )
    """)

# Create loyalty members table
cursor.execute("""CREATE TABLE IF NOT EXISTS loyalty_members (
    member_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_name TEXT UNIQUE NOT NULL,
    tier TEXT NOT NULL,
    points INTEGER DEFAULT 0,
    amount_spent REAL DEFAULT 0,
    joined_date TEXT NOT NULL
    )
    """)

# Create loyalty offers table
cursor.execute("""CREATE TABLE IF NOT EXISTS loyalty_offers (
    offer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_name TEXT NOT NULL,
    description TEXT NOT NULL,
    discount_percentage REAL NOT NULL,
    valid_for_tier TEXT NOT NULL,
    active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL
    )
    """)

# Create stay duration table for tracking receptionist hours
cursor.execute("""CREATE TABLE IF NOT EXISTS receptionist_shifts (
    shift_id INTEGER PRIMARY KEY AUTOINCREMENT,
    receptionist_name TEXT NOT NULL,
    check_in_time TEXT NOT NULL,
    check_out_time TEXT,
    duration_hours INTEGER DEFAULT 0
    )
    """)

conn.commit()

# function to get current time
def get_current_time():
    return strftime("%Y-%m-%d %H:%M:%S")

def get_iso_timestamp():
    """Generate ISO format timestamp without milliseconds"""
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

# function to generate a unique booking ID using UUID
def generate_booking_id():
    """Generate a 6-digit numeric booking ID"""

    # Generate random 6-digit number
    booking_id = str(random.randint(100000, 999999))
    # Ensure it's unique in database
    while True:
        cursor.execute("SELECT booking_id FROM guest_bookings WHERE booking_id = ?", (booking_id,))
        if not cursor.fetchone():
            return booking_id
        booking_id = str(random.randint(100000, 999999))

# function to calculate total amount for a booking
def calculate_total_amount(room_price, check_in_date, check_out_date):
    date_format = "%d-%m-%Y"
    d1 = datetime.datetime.strptime(check_in_date, date_format)
    d2 = datetime.datetime.strptime(check_out_date, date_format)
    delta = d2 - d1
    total_days = delta.days
    if total_days <= 0:
        total_days = 1  # Minimum one day charge
    total_amount = room_price * total_days
    return total_amount

# function to show error messages
def show_error_message(message):
    messagebox.showerror("Error", message)

# function to show info messages
def show_info_message(message):
    messagebox.showinfo("Info", message)
    
# function to show success messages
def show_success_message(message):
    messagebox.showinfo("Success", message)
    
# function to show warning messages
def show_warning_message(message):
    messagebox.showwarning("Warning", message)
    
# function to confirm actions
def confirm_action(message):
    return messagebox.askyesno("Confirm", message)

# function to connect to the database
def connect_to_database():
    return sqlite3.connect("hotel_management.db")

# function to get a database cursor
def get_database_cursor(connection):
    return connection.cursor()

# function to close the database connecting
def close_database_connection(connection):
    connection.close()

# ===== GUEST CRUD FUNCTIONS =====
def get_guest(guest_name):
    """Retrieve guest details by name"""
    try:
        cursor.execute("SELECT * FROM guest_bookings WHERE guest_name = ?", (guest_name,))
        return cursor.fetchone()
    except Exception as e:
        show_error_message(f"Error retrieving guest: {str(e)}")
        return None

def update_guest(guest_name, room_number, check_in_date, check_out_date, status, total_amount):
    """Update guest booking information"""
    try:
        cursor.execute("""UPDATE guest_bookings SET room_number=?, check_in_date=?, check_out_date=?, status=?, total_amount=? 
                         WHERE guest_name=?""", 
                       (room_number, check_in_date, check_out_date, status, total_amount, guest_name))
        conn.commit()
        show_success_message("Guest information updated successfully")
        return True
    except Exception as e:
        show_error_message(f"Error updating guest: {str(e)}")
        return False

def delete_guest(guest_name):
    """Delete a guest profile"""
    try:
        cursor.execute("DELETE FROM guests WHERE guest_name = ?", (guest_name,))
        conn.commit()
        show_success_message("Guest deleted successfully")
        return True
    except Exception as e:
        show_error_message(f"Error deleting guest: {str(e)}")
        return False

def get_all_guests():
    """Retrieve all guest profiles"""
    try:
        cursor.execute("SELECT * FROM guests ORDER BY created_at DESC")
        return cursor.fetchall()
    except Exception as e:
        show_error_message(f"Error retrieving guests: {str(e)}")
        return []

def set_current_user(username, role):
    global current_user, current_user_role
    current_user = username
    current_user_role = role.lower()
    # update UI
    try:
        role_label.config(text=current_user_role.upper())
    except Exception:
        pass
    img = load_profile_image(username)
    if img:
        profile_img_label.config(image=img)
        profile_img_label.image = img

def load_profile_image(username):
    # attempt to load profile image from profiles/<username>.(png|jpg)
    paths = [f"profiles/{username}.png", f"profiles/{username}.jpg", f"profiles/{username}.jpeg"]
    for p in paths:
        if os.path.exists(p):
            try:
                im = Image.open(p).resize((40,40))
                return ImageTk.PhotoImage(im)
            except Exception:
                break
    # fallback: use default logo if available
    try:
        im = logo_image.resize((40,40))
        return ImageTk.PhotoImage(im)
    except Exception:
        return None

def ensure_profiles_dir():
    pdir = os.path.join(os.getcwd(), "profiles")
    if not os.path.exists(pdir):
        os.makedirs(pdir, exist_ok=True)
    # create placeholder if missing
    placeholder = os.path.join(pdir, "placeholder.png")
    if not os.path.exists(placeholder):
        try:
            im = Image.new("RGB", (40,40), color=(200,200,200))
            im.save(placeholder)
        except Exception:
            pass
    return pdir

def upload_profile_image():
    if not current_user:
        show_error_message("No user logged in")
        return
    pdir = ensure_profiles_dir()
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not path:
        return
    dest = os.path.join(pdir, f"{current_user}.png")
    try:
        im = Image.open(path).resize((40,40))
        im.save(dest)
        img = ImageTk.PhotoImage(im)
        profile_img_label.config(image=img)
        profile_img_label.image = img
        show_success_message("Profile image updated")
    except Exception as e:
        show_error_message(f"Error updating profile image: {e}")


def register_window(win):
    """Register a Toplevel so we can close it on logout."""
    try:
        open_windows.append(win)
    except Exception:
        pass
    # apply logo icon and styling
    try:
        if 'logo_image' in globals() and logo_image:
            win_logo = ImageTk.PhotoImage(logo_image)
            win.iconphoto(False, win_logo)
            win.logo_ref = win_logo  # keep reference
    except Exception:
        pass
    def _on_close():
        try:
            open_windows.remove(win)
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
    try:
        win.protocol("WM_DELETE_WINDOW", _on_close)
    except Exception:
        pass


def close_all_toplevels():
    """Close and clear all registered Toplevel windows."""
    while open_windows:
        w = open_windows.pop()
        try:
            w.destroy()
        except Exception:
            pass

def set_user_online(username):
    now = datetime.datetime.utcnow().isoformat()
    try:
        cursor.execute("UPDATE users SET last_active = ? WHERE username = ?", (now, username))
        conn.commit()
    except Exception:
        pass

def set_user_offline(username):
    now = datetime.datetime.utcnow().isoformat()
    try:
        cursor.execute("UPDATE users SET last_active = ? WHERE username = ?", (now, username))
        conn.commit()
    except Exception:
        pass

def get_user_last_active(username):
    try:
        cursor.execute("SELECT last_active FROM users WHERE username = ?", (username,))
        r = cursor.fetchone()
        if r and r[0]:
            return datetime.datetime.fromisoformat(r[0])
    except Exception:
        pass
    return None

def get_relative_time(dt):
    if not dt:
        return "never"
    now = datetime.datetime.utcnow()
    delta = now - dt
    seconds = int(delta.total_seconds())
    minutes = seconds // 60
    if minutes < 1:
        return f"{seconds}s ago"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    years = months // 12
    return f"{years}y ago"

def get_conversation(user1, user2):
    try:
        cursor.execute("SELECT sender, receiver, message, timestamp FROM messages WHERE (sender=? AND receiver=?) OR (receiver=? AND sender=?) ORDER BY timestamp ASC", (user1, user2, user2, user1))
        return cursor.fetchall()
    except Exception:
        return []

def create_notification(username, message_text):
    ts = get_iso_timestamp()
    try:
        # Use a fresh DB connection to avoid threading issues
        conn_local = connect_to_database()
        cur_local = conn_local.cursor()
        cur_local.execute("INSERT INTO notifications (username, message, created_at) VALUES (?, ?, ?)", (username, message_text, ts))
        conn_local.commit()
        conn_local.close()
        # increment badge if current user
        if username == current_user:
            notif_count_var.set(notif_count_var.get() + 1)
    except Exception:
        pass

def notify_invoice_created(booking_id, amount):
    """Notify admin of new invoice"""
    message = f"💳 Invoice Created: Booking {booking_id} - Amount: {amount}"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'admin'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_review_submitted(guest_name, rating):
    """Notify admin of review"""
    message = f"⭐ Review Submitted: {guest_name} - Rating: {rating}/5"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'admin'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_housekeeping_task(task_type, room_number):
    """Notify housekeeping staff"""
    message = f"🧹 Housekeeping Task: {task_type} - Room {room_number}"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'staff'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_maintenance_issue(issue_type, location):
    """Notify maintenance staff"""
    message = f"🔧 Maintenance Issue: {issue_type} - {location}"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'staff'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_restaurant_order(room_number, items):
    """Notify restaurant staff of room service order"""
    message = f"🍽️ Room Service Order: Room {room_number} - {items[:50]}"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'staff'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_loyalty_activity(member_name, points):
    """Notify admin of loyalty program activity"""
    message = f"🎁 Loyalty Program: {member_name} earned {points} points"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'admin'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_password_change(username):
    """Notify user of password change"""
    message = f"🔐 Password Changed: Your password was successfully updated"
    create_notification(username, message)

def notify_checkout(guest_name, room_number):
    """Notify admin and staff of guest checkout"""
    message = f"🏨 Guest Checkout: {guest_name} has checked out from Room {room_number}"
    try:
        cursor.execute("SELECT username FROM users WHERE role IN ('admin', 'staff')")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_new_booking(guest_name, room_number, check_in, check_out):
    """Notify staff and receptionist of new booking"""
    message = f"📅 New Booking: {guest_name} - Room {room_number} ({check_in} to {check_out})"
    try:
        cursor.execute("SELECT username FROM users WHERE role IN ('admin', 'staff', 'receptionist')")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_guest_added(guest_name, room_number):
    """Notify admin of new guest"""
    message = f"👥 Guest Added: {guest_name} in Room {room_number}"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'admin'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_invoice_created(booking_id, amount):
    """Notify admin of new invoice"""
    message = f"💳 Invoice Created: Booking {booking_id} - Amount: {amount}"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'admin'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_review_submitted(guest_name, rating):
    """Notify admin of review"""
    message = f"⭐ Review Submitted: {guest_name} - Rating: {rating}/5"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'admin'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_housekeeping_task(task_type, room_number):
    """Notify housekeeping staff"""
    message = f"🧹 Housekeeping Task: {task_type} - Room {room_number}"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'staff'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_maintenance_issue(issue_type, location):
    """Notify maintenance staff"""
    message = f"🔧 Maintenance Issue: {issue_type} - {location}"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'staff'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_restaurant_order(room_number, items):
    """Notify restaurant staff of room service order"""
    message = f"🍽️ Room Service Order: Room {room_number} - {items[:50]}"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'staff'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_loyalty_activity(member_name, points):
    """Notify admin of loyalty program activity"""
    message = f"🎁 Loyalty Program: {member_name} earned {points} points"
    try:
        cursor.execute("SELECT username FROM users WHERE role = 'admin'")
        for user in cursor.fetchall():
            create_notification(user[0], message)
    except Exception:
        pass

def notify_password_change(username):
    """Notify user of password change"""
    message = f"🔐 Password Changed: Your password was successfully updated"
    create_notification(username, message)

def get_unread_notifications(username):
    try:
        cursor.execute("SELECT id, message, created_at FROM notifications WHERE username=? AND is_read=0 ORDER BY created_at DESC", (username,))
        return cursor.fetchall()
    except Exception:
        return []

def show_notifications_window():
    nw = tk.Toplevel(root)
    register_window(nw)
    nw.title("🔔 Notifications")
    nw.geometry("700x600")
    nw.configure(bg="#f4f6f8")
    
    # Mark all notifications as read immediately when window opens
    try:
        cursor.execute("UPDATE notifications SET is_read=1 WHERE username=?", (current_user,))
        conn.commit()
        # Reset notification count badge
        notif_count_var.set(0)
    except Exception:
        pass
    
    # Header with refresh button
    header_frame = tk.Frame(nw, bg="#112439")
    header_frame.pack(fill="x", padx=0, pady=0)
    
    tk.Label(header_frame, text="Notifications", font=("segoe ui", 14, "bold"), 
             bg="#112439", fg=COLOR_GOLD).pack(side="left", padx=15, pady=10)
    
    def refresh_notifications():
        # Clear and reload
        for widget in notif_list_frame.winfo_children():
            widget.destroy()
        load_all_notifications()
    
    tk.Button(header_frame, text="🔄 Refresh", command=refresh_notifications, 
              bg="#2196F3", fg="white", font=("segoe ui", 10), relief="flat", 
              cursor="hand2", padx=10).pack(side="right", padx=10, pady=10)
    
    # Main notifications list with scrollbar
    canvas = tk.Canvas(nw, bg="#f4f6f8", highlightthickness=0)
    scrollbar = tk.Scrollbar(nw, orient="vertical", command=canvas.yview)
    notif_list_frame = tk.Frame(canvas, bg="#f4f6f8")
    
    notif_list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=notif_list_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    scrollbar.pack(side="right", fill="y")
    
    # Add mousewheel scrolling support
    def _on_mousewheel_notif(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind_all("<MouseWheel>", _on_mousewheel_notif)
    
    def delete_notification(notif_id):
        try:
            cursor.execute("DELETE FROM notifications WHERE id=?", (notif_id,))
            conn.commit()
            refresh_notifications()
        except Exception as e:
            show_error_message(f"Error: {str(e)}")
    
    def mark_as_read(notif_id):
        try:
            cursor.execute("UPDATE notifications SET is_read=1 WHERE id=?", (notif_id,))
            conn.commit()
            refresh_notifications()
        except Exception:
            pass
    
    def load_all_notifications():
        try:
            # Filter out chat messages - only show system notifications (not from 💬 messages)
            cursor.execute("SELECT id, message, created_at, is_read FROM notifications WHERE username=? AND message NOT LIKE '💬%' ORDER BY created_at DESC", (current_user,))
            notifs = cursor.fetchall()
            
            if not notifs:
                tk.Label(notif_list_frame, text="No notifications", font=("segoe ui", 11), 
                        bg="#f4f6f8", fg="#999999").pack(pady=20)
                return
            
            for notif_id, msg, created_at, is_read in notifs:
                # Parse timestamp without milliseconds
                try:
                    dt = datetime.datetime.fromisoformat(created_at)
                    ts_display = dt.strftime("%H:%M:%S")
                except Exception:
                    ts_display = created_at
                
                # Create notification card with border
                notif_card = tk.Frame(notif_list_frame, bg="white", relief="solid", bd=1, highlightthickness=1)
                notif_card.pack(fill="x", pady=6, padx=5)
                
                # Content area
                content_frame = tk.Frame(notif_card, bg="white")
                content_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                # Message with bold text
                msg_label = tk.Label(content_frame, text=msg, font=("segoe ui", 10, "bold"), 
                                    bg="white", justify="left", wraplength=450)
                msg_label.pack(anchor="w", pady=(0, 5))
                
                # Timestamp and read status
                info_frame = tk.Frame(content_frame, bg="white")
                info_frame.pack(fill="x")
                
                read_status = "✓" if is_read else "0"
                read_color = "#999999" if is_read else "#FF5722"
                read_font = ("segoe ui", 9, "italic")
                
                tk.Label(info_frame, text=f"📅 {ts_display}", font=read_font, 
                        bg="white", fg="#666666").pack(side="left", padx=(0, 20))
                
                tk.Label(info_frame, text=f"Status: {read_status}", font=("segoe ui", 9, "bold"), 
                        bg="white", fg=read_color).pack(side="left", padx=(0, 20))
                
                # Action buttons
                btn_frame = tk.Frame(content_frame, bg="white")
                btn_frame.pack(fill="x", pady=(10, 0))
                
                if not is_read:
                    tk.Button(btn_frame, text="Mark Read", command=lambda nid=notif_id: mark_as_read(nid),
                             bg="#4CAF50", fg="white", font=("segoe ui", 9), relief="flat",
                             cursor="hand2", padx=8, pady=3).pack(side="left", padx=(0, 5))
                
                tk.Button(btn_frame, text="Delete", command=lambda nid=notif_id: delete_notification(nid),
                         bg="#f44336", fg="white", font=("segoe ui", 9), relief="flat",
                         cursor="hand2", padx=8, pady=3).pack(side="left")
        
        except Exception as e:
            tk.Label(notif_list_frame, text=f"Error: {str(e)}", font=("segoe ui", 10),
                    bg="#f4f6f8", fg="red").pack(pady=20)
    
    load_all_notifications()


def get_recent_users():
    """Get list of recently logged in users (limit 5)"""
    try:
        recent_file = os.path.join(os.getcwd(), "recent_users.txt")
        if os.path.exists(recent_file):
            with open(recent_file, 'r') as f:
                users = [u.strip() for u in f.readlines() if u.strip()]
                return users[:5]  # return last 5
    except Exception:
        pass
    return []


def save_recent_user(username):
    """Save username to recent logins file"""
    try:
        recent_file = os.path.join(os.getcwd(), "recent_users.txt")
        users = get_recent_users()
        # remove if already exists, then add to front
        if username in users:
            users.remove(username)
        users.insert(0, username)
        # keep only last 5
        users = users[:5]
        with open(recent_file, 'w') as f:
            f.write('\n'.join(users))
    except Exception:
        pass
# face recognition window
def face_recognition_window():
    """Comprehensive Face Recognition window - ALL face features in one place"""
    face_win = tk.Toplevel(root)
    register_window(face_win)
    face_win.title("Face Recognition - Grand Suit Hotels")
    face_win.geometry("600x700")
    face_win.resizable(False, False)
    face_win.config(bg="#470ca0")
    
    # center window
    face_win.update_idletasks()
    width = 600
    height = 700
    x = (face_win.winfo_screenwidth() // 2) - (width // 2)
    y = (face_win.winfo_screenheight() // 2) - (height // 2)
    face_win.geometry(f"{width}x{height}+{x}+{y}")
    
    # Load logo
    try:
        face_logo = ImageTk.PhotoImage(logo_image)
        face_win.iconphoto(True, face_logo)
    except Exception:
        pass
    
    # ===== HEADER SECTION =====
    tk.Label(
        face_win,
        text="🏨 GRAND SUIT HOTELS",
        font=("segoe ui", 16, "bold"),
        bg="#470ca0",
        fg="#f59e0b"
    ).pack(pady=10)
    
    tk.Label(
        face_win,
        text="Face Recognition System",
        font=("segoe ui", 13, "bold"),
        bg="#470ca0",
        fg="white"
    ).pack(pady=3)
    
    # ===== FACE VERIFICATION SECTION =====
    tk.Label(
        face_win,
        text="─────  Face Login  ─────",
        font=("segoe ui", 10, "bold"),
        bg="#470ca0",
        fg="#FFD700"
    ).pack(pady=10)
    
    tk.Label(
        face_win,
        text="Click face button to login",
        font=("segoe ui", 9),
        bg="#470ca0",
        fg="white"
    ).pack(pady=2)
    
    # Status label
    status_label = tk.Label(
        face_win,
        text="",
        font=("segoe ui", 9, "bold"),
        bg="#470ca0",
        fg="white"
    )
    status_label.pack(pady=8)
    
    # Verification state
        # Verification state
    verify_state = {"in_progress": False, "found": False, "failed": False, "failed_options_shown": False}
    
    # start the verivication process
    def start_verification():
        """Start face verification for login"""
        if verify_state["in_progress"]:
            return
        
        # Reset failed options flag so we can show try again button on next failure
        verify_state["failed_options_shown"] = False
        verify_state["failed"] = False
        verify_state["found"] = False
        
        if not FACE_RECOGNITION_AVAILABLE:
            show_error_message("OpenCV not installed. Using manual login instead.")
            face_win.destroy()
            login_window()
            return
        
        verify_state["in_progress"] = True
        verify_state["failed"] = False  # Reset failed state
        verify_btn.config(state="disabled", relief="sunken")
        status_label.config(text="🔍 Scanning...", fg="#FFD700")
        face_win.update()
        
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                status_label.config(text="⚠ Webcam not available", fg="#FF9800")
                face_win.update()
                face_win.after(2000, lambda: verify_btn.config(state="normal", relief="raised"))
                verify_state["in_progress"] = False
                return
            
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            start_time = datetime.datetime.now()
            timeout = 5
            
            cv2.namedWindow('Face Login - Hold Still', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Face Login - Hold Still', 640, 480)
            
            while (datetime.datetime.now() - start_time).total_seconds() < timeout:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                elapsed = int((datetime.datetime.now() - start_time).total_seconds())
                cv2.putText(frame, f"Time: {elapsed}s / 5s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if len(faces) > 0:
                    cv2.putText(frame, "Face detected! Analyzing...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    (x, y, w, h) = faces[0]
                    face_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    current_encoding = np.mean(face_roi)
                    
                    best_match = None
                    best_distance = float('inf')
                    
                    ensure_face_db_dir()
                    if os.path.exists(FACE_DB_DIR):
                        for user_file in os.listdir(FACE_DB_DIR):
                            if user_file.endswith('_face.npy'):
                                username = user_file.replace('_face.npy', '')
                                try:
                                    stored_encoding = np.load(os.path.join(FACE_DB_DIR, user_file))
                                    distance = abs(current_encoding - stored_encoding)
                                    
                                    if distance < 50 and distance < best_distance:
                                        best_match = username
                                        best_distance = distance
                                except Exception:
                                    pass
                    
                    if best_match:
                        cv2.putText(frame, f"Match found: {best_match}!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow('Face Login - Hold Still', frame)
                        cv2.waitKey(1)
                        
                        verify_state["found"] = True
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        status_label.config(text=f"✓ Verified: {best_match}!", fg="#4CAF50")
                        face_win.update()
                        
                        face_win.after(2000, lambda: auto_login_user(best_match, face_win))
                        return
                
                cv2.imshow('Face Login - Hold Still', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            if not verify_state["found"]:
                # Mark as failed and only show options once
                if not verify_state.get("failed"):
                    verify_state["failed"] = True
                    status_label.config(text="✗ Face not verified\nTry again or use manual login", fg="#f44336")
                    face_win.update()
                    show_failed_options()
            
        except Exception as e:
            print(f"Verification error: {e}")
            status_label.config(text="⚠ Verification error", fg="#FF9800")
            face_win.update()
            if not verify_state.get("failed"):
                verify_state["failed"] = True
                show_failed_options()
        finally:
            verify_state["in_progress"] = False
    
    def show_failed_options():
        """Show options when face verification fails - shown only once"""
        # Only show if we haven't already shown failed options
        if verify_state.get("failed_options_shown"):
            return
        verify_state["failed_options_shown"] = True
        
        try:
            verify_btn.pack_forget()
        except Exception:
            pass
        
        tk.Button(
            face_win,
            text="😐  TRY AGAIN",
            font=("Arial", 15, "bold"),
            bg="#f59e0b",
            fg="#000000",
            relief="raised",
            bd=3,
            cursor="hand2",
            command=start_verification,
            padx=20,
            pady=8
        ).pack(pady=8)
        
        tk.Label(
            face_win,
            text="─────────────────",
            font=("segoe ui", 9),
            bg="#470ca0",
            fg="#FFD700"
        ).pack(pady=5)
    
    def auto_login_user(username, win):
        """Auto-login with verified face"""
        try:
            conn_temp = connect_to_database()
            cur_temp = get_database_cursor(conn_temp)
            cur_temp.execute("SELECT password, role FROM users WHERE username = ?", (username,))
            row = cur_temp.fetchone()
            close_database_connection(conn_temp)
            
            if row:
                role = row[1]
                set_current_user(username, role)
                set_user_online(username)
                save_recent_user(username)
                apply_role_permission()
                
                try:
                    cur_temp = connect_to_database().cursor()
                    cur_temp.execute("SELECT value FROM settings WHERE key='auto_refresh_kpis'")
                    v = cur_temp.fetchone()
                    if v and v[0] == '1':
                        start_kpi_refresh()
                except Exception:
                    pass
                
                unread = len(get_unread_notifications(username))
                notif_count_var.set(unread)
                show_success_message(f"Welcome back, {username}!")
                win.destroy()
                root.deiconify()
                
                try:
                    start_polling()
                except Exception:
                    pass
        except Exception as e:
            show_error_message(f"Login error: {str(e)}")
    
    # Face emoji button - COMPACT
    verify_btn = tk.Button(
        face_win,
        text="😐",
        font=("Arial", 15, "bold"),
        bg="#f59e0b",
        fg="#000000",
        relief="raised",
        bd=5,
        cursor="hand2",
        command=start_verification,
        width=6,
        height=1
    )
    verify_btn.pack(pady=12)
    
    # ===== FACE REGISTRATION SECTION =====
    tk.Label(
        face_win,
        text="─────  Face Registration  ─────",
        font=("segoe ui", 10, "bold"),
        bg="#470ca0",
        fg="#FFD700"
    ).pack(pady=8)
    
    tk.Label(
        face_win,
        text="📷 Register Your Face",
        font=("segoe ui", 11, "bold"),
        bg="#470ca0",
        fg="white"
    ).pack(pady=2)
    
    tk.Label(
        face_win,
        text="Enter username & password",
        font=("segoe ui", 8),
        bg="#470ca0",
        fg="#cccccc"
    ).pack(pady=1)
    
    def open_registration():
        """Open registration in a modal window"""
        reg_win = tk.Toplevel(face_win)
        register_window(reg_win)
        reg_win.title("Face Registration")
        reg_win.geometry("300x400")
        reg_win.resizable(False, False)
        reg_win.config(bg="#470ca0")
        
        # center window
        reg_win.update_idletasks()
        w = 300
        h = 400
        x_pos = (reg_win.winfo_screenwidth() // 2) - (w // 2)
        y_pos = (reg_win.winfo_screenheight() // 2) - (h // 2)
        reg_win.geometry(f"{w}x{h}+{x_pos}+{y_pos}")
        
        tk.Label(
            reg_win,
            text="Face Registration",
            font=("segoe ui", 14, "bold"),
            bg="#470ca0",
            fg="#f59e0b"
        ).pack(pady=20)
        
        tk.Label(
            reg_win,
            text="Username:",
            bg="#989898",
            fg="#1a1a1a",
            font=("segoe ui", 11, "bold")
        ).pack(ipady=5, pady=5)
        
        username_entry = tk.Entry(
            reg_win,
            bg="#ffffff",
            fg="#1a1a1a",
            font=("segoe ui", 10, "bold"),
            relief="solid",
            bd=2,
            width=35
        )
        username_entry.pack(ipadx=10, ipady=6, pady=5)
        
        tk.Label(
            reg_win,
            text="Password:",
            bg="#989898",
            fg="#1a1a1a",
            font=("segoe ui", 11, "bold")
        ).pack(ipady=5, pady=5)
        
        password_entry = tk.Entry(
            reg_win,
            show="*",
            bg="#ffffff",
            fg="#1a1a1a",
            font=("segoe ui", 10, "bold"),
            relief="solid",
            bd=2,
            width=35
        )
        password_entry.pack(ipadx=10, ipady=6, pady=5)
        
        def register_face():
            """Register user's face"""
            username = username_entry.get().strip()
            password = password_entry.get().strip()
            
            if not username or not password:
                show_error_message("Please enter username and password")
                return
            
            try:
                conn_temp = connect_to_database()
                cur_temp = get_database_cursor(conn_temp)
                cur_temp.execute("SELECT password, role FROM users WHERE username = ?", (username,))
                row = cur_temp.fetchone()
                close_database_connection(conn_temp)
                
                if not row:
                    show_error_message("User does not exist")
                    return
                
                if not verify_password(password, row[0]):
                    show_error_message("Incorrect password")
                    return
                
            except Exception as e:
                show_error_message(f"Error: {str(e)}")
                return
            
            if capture_face_encoding(username):
                show_success_message(f"Face registered for {username}!")
                reg_win.destroy()
            else:
                show_error_message("Face registration failed")
        
        tk.Button(
            reg_win,
            text="📷  REGISTER",
            font=("segoe ui", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            relief="raised",
            bd=3,
            cursor="hand2",
            command=register_face,
            padx=20,
            pady=10
        ).pack(pady=15, padx=20, fill="x")
    
    tk.Button(
        face_win,
        text="📷  REGISTER FACE",
        font=("segoe ui", 11, "bold"),
        bg="#4CAF50",
        fg="white",
        relief="raised",
        bd=2,
        cursor="hand2",
        command=open_registration,
        padx=15,
        pady=8
    ).pack(pady=6, padx=25, fill="x")
    
    # ===== ACTION BUTTONS =====
    tk.Label(
        face_win,
        text="─────────  Options  ─────────",
        font=("segoe ui", 10, "bold"),
        bg="#470ca0",
        fg="#FFD700"
    ).pack(pady=8)
    
    button_frame = tk.Frame(face_win, bg="#470ca0")
    button_frame.pack(pady=8, padx=20, fill="x")
    
    tk.Button(
        button_frame,
        text="🔐  MANUAL LOGIN",
        font=("segoe ui", 10, "bold"),
        bg="#d4a574",
        fg="#1a1a1a",
        relief="raised",
        bd=2,
        cursor="hand2",
        command=lambda: (face_win.destroy(), login_window()),
        padx=12,
        pady=8
    ).pack(side="left", padx=8, fill="x", expand=True)
    
    tk.Button(
        button_frame,
        text="👥  RECENT USERS",
        font=("segoe ui", 10, "bold"),
        bg="#FF9800",
        fg="white",
        relief="raised",
        bd=2,
        cursor="hand2",
        command=lambda: (face_win.destroy(), recent_users_window()),
        padx=12,
        pady=8
    ).pack(side="left", padx=8, fill="x", expand=True)
    
    # Footer
    tk.Label(
        face_win,
        text="Secure Hotel Management System",
        font=("segoe ui", 8),
        bg="#470ca0",
        fg="#cccccc"
    ).pack(pady=10)

def login_window(prefill_username=None):
    """Clean login window - Username, Password, Signup"""
    login_win = tk.Toplevel(root)
    register_window(login_win)
    login_win.title("Manual Login - Grand Suit Hotels")
    login_win.geometry("600x600")
    login_win.resizable(False, False)
    login_win.config(bg="#470ca0")

    # center login window
    login_win.update_idletasks()
    width = 600
    height = 600
    x = (login_win.winfo_screenwidth() // 2) - (width // 2)
    y = (login_win.winfo_screenheight() // 2) - (height // 2)
    login_win.geometry(f"{width}x{height}+{x}+{y}")

    # Load logo image safely
    try:
        login_logo = ImageTk.PhotoImage(logo_image)
        login_win.iconphoto(True, login_logo)
    except Exception:
        pass
    
    # ===== HEADER SECTION =====
    tk.Label(
        login_win,
        text="🏨 GRAND SUIT HOTELS",
        font=("segoe ui", 16, "bold"),
        bg="#470ca0",
        fg="#f59e0b"
    ).pack(pady=10)
    
    tk.Label(
        login_win,
        text="Manual Login",
        font=("segoe ui", 13, "bold"),
        bg="#470ca0",
        fg="white"
    ).pack(pady=3)
    
    # ===== RECENT USERS SECTION =====
    recent_users = get_recent_users()
    if recent_users:
        tk.Label(
            login_win,
            text="─────  Recent Users  ─────",
            font=("segoe ui", 9, "bold"),
            bg="#470ca0",
            fg="#FFD700"
        ).pack(pady=8)
        
        def open_recent_users():
            login_win.destroy()
            recent_users_window()
        
        tk.Button(
            login_win,
            text="👥  Recent Users",
            font=("segoe ui", 10, "bold"),
            bg="#FF9800",
            fg="white",
            relief="raised",
            bd=2,
            cursor="hand2",
            command=open_recent_users,
            padx=15,
            pady=7
        ).pack(pady=4, padx=25, fill="x")
        
        tk.Label(
            login_win,
            text="or enter below",
            font=("segoe ui", 8),
            bg="#470ca0",
            fg="#aaaaaa"
        ).pack(pady=2)

    # ===== CREDENTIALS SECTION =====
    tk.Label(
        login_win,
        text="─────  Credentials  ─────",
        font=("segoe ui", 9, "bold"),
        bg="#470ca0",
        fg="#FFD700"
    ).pack(pady=10)

    tk.Label(
        login_win,
        text="Username:",
        bg="#989898",
        fg="#1a1a1a",
        font=("segoe ui", 10, "bold")
    ).pack(ipady=4, pady=6)
    
    username_entry = tk.Entry(
        login_win,
        bg="#ffffff",
        fg="#1a1a1a",
        font=("segoe ui", 10, "bold"),
        relief="solid",
        bd=2,
        width=35
    )
    username_entry.pack(ipadx=12, ipady=6, pady=3)

    tk.Label(
        login_win,
        text="Password:",
        bg="#989898",
        fg="#1a1a1a",
        font=("segoe ui", 10, "bold")
    ).pack(ipady=4, pady=6)
    
    password_entry = tk.Entry(
        login_win,
        show="*",
        bg="#ffffff",
        fg="#1a1a1a",
        font=("segoe ui", 10, "bold"),
        relief="solid",
        bd=2,
        width=35
    )
    password_entry.pack(ipadx=12, ipady=6, pady=3)
    
    # Forgotten password link
    def open_forgotten_password():
        """Open forgotten password window"""
        fpw = tk.Toplevel(login_win)
        register_window(fpw)
        fpw.title("Reset Password")
        fpw.geometry("450x400")
        fpw.configure(bg="#E6E1E1")
        fpw.resizable(False, False)
        
        # Center window
        fpw.update_idletasks()
        w = 450
        h = 400
        x = (fpw.winfo_screenwidth() // 2) - (w // 2)
        y = (fpw.winfo_screenheight() // 2) - (h // 2)
        fpw.geometry(f"{w}x{h}+{x}+{y}")
        
        tk.Label(fpw, text="Reset Your Password", font=("segoe ui", 14, "bold"), bg="#E6E1E1", fg="#1a3a52").pack(pady=15)
        
        tk.Label(fpw, text="Enter your username:", font=("segoe ui", 10), bg="#E6E1E1").pack(pady=10)
        user_entry = tk.Entry(fpw, font=("segoe ui", 10), bg="white", width=35)
        user_entry.pack(padx=20, pady=5)
        
        tk.Label(fpw, text="New Password:", font=("segoe ui", 10), bg="#E6E1E1").pack(pady=(15, 5))
        pwd_entry = tk.Entry(fpw, show="*", font=("segoe ui", 10), bg="white", width=35)
        pwd_entry.pack(padx=20, pady=5)
        
        strength_label = tk.Label(fpw, text="", font=("segoe ui", 8), bg="#E6E1E1")
        strength_label.pack(pady=2)
        
        def check_strength(*args):
            pwd = pwd_entry.get()
            if pwd:
                is_valid, msg = validate_password_strength(pwd)
                if is_valid:
                    strength_label.config(text="✓ " + msg, fg="#2E7D32")
                else:
                    strength_label.config(text="✗ " + msg, fg="#C62828")
            else:
                strength_label.config(text="")
        
        pwd_entry.bind("<KeyRelease>", check_strength)
        
        tk.Label(fpw, text="Confirm Password:", font=("segoe ui", 10), bg="#E6E1E1").pack(pady=(15, 5))
        pwd_confirm = tk.Entry(fpw, show="*", font=("segoe ui", 10), bg="white", width=35)
        pwd_confirm.pack(padx=20, pady=5)
        
        def reset_password():
            username = user_entry.get().strip()
            new_pwd = pwd_entry.get()
            conf_pwd = pwd_confirm.get()
            
            if not all([username, new_pwd, conf_pwd]):
                show_error_message("All fields are required")
                return
            
            if new_pwd != conf_pwd:
                show_error_message("Passwords do not match")
                return
            
            if len(new_pwd) < 8:
                show_error_message("Password must be at least 8 characters")
                return
            
            try:
                cursor.execute("SELECT username FROM users WHERE username=?", (username,))
                if not cursor.fetchone():
                    show_error_message("Username not found")
                    return
                
                new_hash = hash_password(new_pwd)
                cursor.execute("UPDATE users SET password=? WHERE username=?", (new_hash, username))
                conn.commit()
                show_success_message("Password reset successful! You can now login.")
                fpw.destroy()
            except Exception as e:
                show_error_message(f"Error resetting password: {e}")
        
        user_entry.bind("<Return>", lambda e: pwd_entry.focus())
        pwd_entry.bind("<Return>", lambda e: pwd_confirm.focus())
        pwd_confirm.bind("<Return>", lambda e: reset_password())
        
        tk.Button(fpw, text="Reset Password", command=reset_password, bg="#2196F3", fg="white", font=("segoe ui", 11, "bold")).pack(pady=15)
    
    forgot_frame = tk.Frame(login_win, bg="#470ca0")
    forgot_frame.pack(pady=3)
    
    tk.Label(forgot_frame, text="Forgot password? ", font=("segoe ui", 9), bg="#470ca0", fg="white").pack(side="left")
    forgot_link = tk.Label(forgot_frame, text="🔐 Click here", font=("segoe ui", 9, "bold", "underline"), bg="#470ca0", fg="#FFD700", cursor="hand2")
    forgot_link.pack(side="left")
    forgot_link.bind("<Button-1>", lambda e: open_forgotten_password())


    # If caller prefills username (recent user window), apply it
    try:
        if prefill_username:
            username_entry.delete(0, 'end')
            username_entry.insert(0, prefill_username)
            password_entry.focus()
    except Exception:
        pass

    def attempt_login():
        """Standard login with username and password"""
        global current_user, current_user_role
        username = username_entry.get().strip()
        password = password_entry.get().strip()
        
        if not username or not password:
            show_error_message("ALL fields required")
            return
        
        conn_temp = connect_to_database()
        cur_temp = get_database_cursor(conn_temp)
        cur_temp.execute("SELECT password, role FROM users WHERE username = ?", (username,))
        row = cur_temp.fetchone()
        user = None
        
        if row:
            stored_pw, role = row
            if verify_password(password, stored_pw):
                user = (role,)
            else:
                try:
                    if password == stored_pw:
                        newh = hash_password(password)
                        cur_temp.execute("UPDATE users SET password=? WHERE username=?", (newh, username))
                        conn_temp.commit()
                        user = (role,)
                except Exception:
                    pass
        
        close_database_connection(conn_temp)
        
        if user:
            role = user[0]
            set_current_user(username, role)
            set_user_online(username)
            save_recent_user(username)
            apply_role_permission()
            
            try:
                cur_temp = connect_to_database().cursor()
                cur_temp.execute("SELECT value FROM settings WHERE key='auto_refresh_kpis'")
                v = cur_temp.fetchone()
                if v and v[0] == '1':
                    start_kpi_refresh()
            except Exception:
                pass
            
            unread = len(get_unread_notifications(username))
            notif_count_var.set(unread)
            show_success_message("Login successful!")
            login_win.destroy()
            root.deiconify()
            
            try:
                start_polling()
            except Exception:
                pass

            # WebSocket chat functionality removed
            pass
        else:
            show_error_message("Invalid username or password.")

    # Bind Enter key to login on both fields
    username_entry.bind("<Return>", lambda e: password_entry.focus())
    password_entry.bind("<Return>", lambda e: attempt_login())

    # ===== BUTTONS SECTION =====
    tk.Label(
        login_win,
        text="─────────  Action  ─────────",
        font=("segoe ui", 9, "bold"),
        bg="#470ca0",
        fg="#FFD700"
    ).pack(pady=10)

    button_frame = tk.Frame(login_win, bg="#470ca0")
    button_frame.pack(pady=8, padx=20, fill="x")

    tk.Button(
        button_frame,
        text="🔓  LOGIN",
        bg="#FFD700",
        fg="#000000",
        font=("segoe ui", 11, "bold"),
        relief="raised",
        bd=2,
        cursor="hand2",
        command=attempt_login,
        padx=15,
        pady=8
    ).pack(side="left", padx=8, fill="x", expand=True)
    
    tk.Button(
        button_frame,
        text="📝  SIGNUP",
        bg="#4CAF50",
        fg="white",
        font=("segoe ui", 11, "bold"),
        relief="raised",
        bd=2,
        cursor="hand2",
        command=signup_window,
        padx=15,
        pady=8
    ).pack(side="left", padx=8, fill="x", expand=True)
    
    # Footer
    tk.Label(
        login_win,
        text="Secure Hotel Management System",
        font=("segoe ui", 7, "italic"),
        bg="#470ca0",
        fg="#888888"
    ).pack(pady=6)

# recent users window
def recent_users_window():
    """Display recent users with profile pictures in a vertically-scrolled window"""
    rw = tk.Toplevel(root)
    register_window(rw)
    rw.title("Recent Users")
    rw.geometry("380x500")
    rw.config(bg=COLOR_SIDEBAR)
    rw.resizable(False, False)

    # Title
    title_frame = tk.Frame(rw, bg=COLOR_GOLD, height=50)
    title_frame.pack(fill='x', side="left")
    title_frame.pack_propagate(False)
    tk.Label(title_frame, text="Recent Users", bg=COLOR_GOLD, fg="#000000", font=("segoe ui", 14, "bold")).pack(pady=10)

    # Scrollable content
    canvas = tk.Canvas(rw, bg=COLOR_SIDEBAR, highlightthickness=0)
    scrollbar = tk.Scrollbar(rw, orient="vertical", command=canvas.yview)
    content = tk.Frame(canvas, bg=COLOR_SIDEBAR)
    
    content.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Mouse wheel support
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind("<MouseWheel>", on_mousewheel)

    users = get_recent_users()
    pdir = ensure_profiles_dir()

    def pick_user(u):
        try:
            rw.destroy()
        except Exception:
            pass
        login_window(prefill_username=u)

    if not users:
        tk.Label(content, text="No recent users", bg=COLOR_SIDEBAR, fg="#999999", font=("segoe ui", 10)).pack(pady=20)
    
    for u in users:
        # User card frame
        card = tk.Frame(content, bg="#1a3a52", relief="flat", bd=0)
        card.pack(fill='x', pady=10, padx=15)
        
        # Inner frame for layout
        inner = tk.Frame(card, bg="#1a3a52")
        inner.pack(fill='x', padx=12, pady=12)
        
        # Profile picture
        img_obj = None
        profile_size = (50, 50)
        
        for ext in ('.png', '.jpg', '.jpeg'):
            p = os.path.join(pdir, f"{u}{ext}")
            if os.path.exists(p):
                try:
                    im = Image.open(p).resize(profile_size, Image.Resampling.LANCZOS)
                    img_obj = ImageTk.PhotoImage(im)
                    break
                except Exception:
                    img_obj = None
        
        if not img_obj and logo_image:
            try:
                im = logo_image.resize(profile_size, Image.Resampling.LANCZOS)
                img_obj = ImageTk.PhotoImage(im)
            except Exception:
                img_obj = None
        
        # Create avatar placeholder if no image
        if not img_obj:
            avatar = tk.Frame(inner, bg=COLOR_GOLD, width=50, height=50)
            avatar.pack(side='left', padx=8)
            avatar.pack_propagate(False)
            avatar_label = tk.Label(avatar, text=u[0].upper(), bg=COLOR_GOLD, fg="#000000", font=("segoe ui", 16, "bold"))
            avatar_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            img_label = tk.Label(inner, image=img_obj, bg="#1a3a52")
            img_label.image = img_obj
            img_label.pack(side='left', padx=8)
        
        # User info
        info_frame = tk.Frame(inner, bg="#1a3a52")
        info_frame.pack(side='left', fill='x', expand=True, padx=10)
        
        tk.Label(info_frame, text=u, bg="#1a3a52", fg="white", font=("segoe ui", 11, "bold"), anchor="w").pack(fill='x')
        tk.Label(info_frame, text="Click to login", bg="#1a3a52", fg="#aaaaaa", font=("segoe ui", 9), anchor="w").pack(fill='x')
        
        # Button
        btn = tk.Button(inner, text="→", bg=COLOR_GOLD, fg="#000000", relief="flat", font=("segoe ui", 12, "bold"), cursor="hand2", command=lambda uu=u: pick_user(uu), width=3)
        btn.pack(side='right', padx=5)


# signup setups
def signup_window():
    signup = tk.Toplevel()
    signup.transient(root)  # Make it modal - stays on top
    signup.grab_set()  # Block interaction with other windows
    register_window(signup)
    signup.title("Sign Up")
    signup.geometry("400x550")
    signup.configure(bg="#E6E1E1")
    signup.resizable(False, False)

    if logo_image:
        try:
            signup_logo = ImageTk.PhotoImage(logo_image)
            signup.iconphoto(False, signup_logo)
        except Exception:
            pass

    tk.Label(
        signup, text="Create a new account",
        font=("segoe", 20, "bold"),
        bg="#C0BEBE", fg="black"
    ).pack(pady=10)

    # signup labels
    tk.Label(signup, text="Username:", font=("segoe ", 10), bg="#c9c9c9").pack(ipady=7, pady=10)
    entry_username = tk.Entry(signup, bg="#c9c9c9", font=("segoe", 10, "bold"))
    entry_username.pack(ipadx=20)

    tk.Label(signup, text="Password:", font=("segoe", 10, "bold"), bg="#c9c9c9").pack(ipady=7, pady=10)
    entry_password = tk.Entry(signup, bg="#c9c9c9", show="*", font=("segoe", 10))
    entry_password.pack(ipadx=20, pady=5)

    strength_label = tk.Label(signup, text="", font=("segoe", 8), bg="#c9c9c9")
    strength_label.pack(pady=2)
    
    def check_pwd_strength(*args):
        pwd = entry_password.get()
        if pwd:
            is_valid, msg = validate_password_strength(pwd)
            if is_valid:
                strength_label.config(text="✓ " + msg, fg="#2E7D32")
            else:
                strength_label.config(text="✗ " + msg, fg="#C62828")
        else:
            strength_label.config(text="")
    
    entry_password.bind("<KeyRelease>", check_pwd_strength)

    tk.Label(signup, text="Confirm password:", font=("segoe", 10, "bold"), bg="#c9c9c9").pack(ipady=7, pady=10)
    entry_confirm_password = tk.Entry(signup, bg="#c9c9c9", show="*", font=("segoe", 10))
    entry_confirm_password.pack(ipadx=20, pady=5)

    tk.Label(signup, text="Role:", font=("segoe", 10, "bold"), bg="#c9c9c9").pack(ipady=7, pady=10)
    role_var = tk.StringVar()
    role_menu = ttk.Combobox(
        signup, background="#c9c9c9", textvariable=role_var, values=["staff", "admin", "receptionist"], state="readonly"
    )
    role_menu.pack(ipadx=5, pady=5)
    
    def create_account():
        username = entry_username.get().strip()
        password = entry_password.get()
        confirm_password = entry_confirm_password.get()
        role = role_var.get()
        
        # Validate input
        if not username or not password or not role:
            show_error_message("Fill all necessary fields")
            return
        
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        if cursor.fetchone():
            show_error_message("Username already exists")
            return
        
        if password != confirm_password:
            show_error_message("Passwords do not match")
            return
        
        # Validate password strength
        is_valid, msg = validate_password_strength(password)
        if not is_valid:
            show_error_message(msg)
            return
        
        # Hash password and create account
        try:
            hpw = hash_password(password)
        except Exception as e:
            show_error_message(f"Error hashing password: {e}")
            return
        
        try:
            cursor.execute("INSERT INTO users(username, password, role) VALUES (?, ?, ?)", (username, hpw, role))
            conn.commit()
            show_success_message("Account created successfully! You can now login.")
            signup.destroy()
        except Exception as e:
            show_error_message(f"Error creating account: {e}")
    
    # bind Enter key for field navigation
    entry_username.bind("<Return>", lambda e: entry_password.focus())
    entry_password.bind("<Return>", lambda e: entry_confirm_password.focus())
    entry_confirm_password.bind("<Return>", lambda e: role_menu.focus())
    role_menu.bind("<Return>", lambda e: create_account())

    tk.Button(signup,
        text="Create Account",
        font=("segoe", 12, "bold"),
        width=16,
        command=create_account,
        bg="#000000",
        fg="#ffffff",
        cursor="hand2",
        relief="flat"
    ).pack(pady=20)

#open the main application window
root = tk.Tk()
root.title("Hotel Management System")
root.geometry("1280x720")
root.resizable(True, True)
root.config(bg="#ffffff")
root.attributes('-disable', False)

# Load logo image globally for access in all windows
global logo_image, logo, logo_photo
logo_image = None
logo = None
logo_photo = None

try:
    logo_image = Image.open("IMG_4898.JPG")
    logo = logo_image
    logo_photo = ImageTk.PhotoImage(logo_image)
    root.iconphoto(False, logo_photo)
except FileNotFoundError:
    pass

# Initially hide the dashboard content and sidebar - only show after successful login
root.withdraw()

# create the bar and frames "top"
top_bar = tk.Frame(root, bg="#ffffff",height=70)
# Don't pack until after login

top_bar_label = tk.Label(top_bar,
                         text="🏨 GRAND SUIT HOTELS",
                         font=("segoe ui",20,"bold"),
                         fg="#1f2a44",bg="#ffffff"
                         )

top_bar_label.pack(side="left",padx=30)

#staff label
# Dynamic role display and profile image (will update on login)
role_frame = tk.Frame(top_bar, bg="#ffffff")
role_frame.pack(side="right", padx=20)

# profile image placeholder
profile_img_label = tk.Label(role_frame, bg="#ffffff")
profile_img_label.pack(side="right", padx=(10,0))
upload_profile_btn = tk.Button(role_frame, text="✎", font=("segoe ui",9), bg="#ffffff", relief="flat", cursor="hand2", command=upload_profile_image)
upload_profile_btn.pack(side="right", padx=(0,6))

# role text label
role_label = tk.Label(role_frame,
                      text="",
                      font=("segoe ui",15, "bold"),
                      fg="#1f2a44",
                      bg="#ffffff")
role_label.pack(side="right")

# notification bell (with badge support)
notif_frame = tk.Frame(top_bar, bg="#ffffff")
notif_frame.pack(side="right", padx=10)
notif_count_var = tk.IntVar(value=0)
notif_button = tk.Button(notif_frame, text="🔔", font=("segoe ui",12), bg="#ffffff", relief="flat", cursor="hand2", command=show_notifications_window)
notif_button.pack(side="left")
notif_badge = tk.Label(notif_frame, textvariable=notif_count_var, bg="#c62828", fg="white", font=("segoe ui",8), width=2)
notif_badge.place(x=18, y=0)

# side bars - initially hidden until login
side_bar_container = tk.Frame(root, bg=COLOR_SIDEBAR, width=220)
# Don't pack yet - will be packed after successful login
side_bar_container.pack_propagate(False)

side_canvas = tk.Canvas(
    side_bar_container,
    bg=COLOR_SIDEBAR,
    highlightthickness=0,
    width=220
)
side_scrollbar = tk.Scrollbar(
    side_bar_container,
    orient="vertical",
    command=side_canvas.yview
)

side_bar = tk.Frame(side_canvas, bg=COLOR_SIDEBAR)

side_bar.bind(
    "<Configure>",
    lambda e: side_canvas.configure(scrollregion=side_canvas.bbox("all"))
)

side_canvas.create_window((0, 0), window=side_bar, anchor="nw")
side_canvas.configure(yscrollcommand=side_scrollbar.set)

side_canvas.pack(side="left", fill="both", expand=True)
# Scrollbar will be shown/hidden based on user role in apply_role_permission()

def sidebar_mousewheel(event):
    side_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

side_canvas.bind("<Enter>", lambda e: side_canvas.bind_all("<MouseWheel>", sidebar_mousewheel))
side_canvas.bind("<Leave>", lambda e: side_canvas.unbind_all("<MouseWheel>"))

def _on_sidebar_mousewheel(event):
    side_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

# menu button creation
menu_buttons = {}
def menu_button(text, roles, command=None):
    btn = tk.Button(
        side_bar,
        text=text,
        font=("segoe ui", 11),
        fg="#ffffff",
        bg=COLOR_SIDEBAR,
        activebackground=COLOR_SIDEBAR_HOVER,
        activeforeground="white",
        relief="flat",
        anchor="w",
        padx=20,
        cursor="hand2",
        command=command
    )
    btn.allowed_roles = roles
    # hover effects
    def _on_enter(e):
        try:
            btn.config(bg=COLOR_SIDEBAR_HOVER, fg="white")
        except Exception:
            pass
    def _on_leave(e):
        try:
            btn.config(bg=COLOR_SIDEBAR, fg="#ffffff")
        except Exception:
            pass
    btn.bind("<Enter>", _on_enter)
    btn.bind("<Leave>", _on_leave)
    menu_buttons[text] = btn
    return btn

def apply_role_permission():
    # pack UI after login
    try:
        top_bar.pack(side="top", fill="x")
        side_bar_container.pack(side="left", fill="y")
        main_content.pack(side="right", fill="both", expand=True)
        root.deiconify()
    except Exception:
        pass
    
    # show/hide buttons based on current user role
    for btn in menu_buttons.values():
        try:
            if current_user_role and current_user_role in btn.allowed_roles:
                btn.pack(fill="x", pady=2)
            else:
                btn.pack_forget()
        except Exception:
            pass
    
    # hide management section for non-admin roles
    try:
        if current_user_role and current_user_role != "admin":
            mgmt_frame.pack_forget()
            # Hide scrollbar for staff and receptionist (not many items)
            side_scrollbar.pack_forget()
        else:
            mgmt_frame.pack(fill="x", pady=(10, 5))
            # Show scrollbar for admin
            side_scrollbar.pack(side="right", fill="y")
    except Exception:
        pass

def get_count(query, params=()):
    conn = connect_to_database()
    cur = conn.cursor()
    cur.execute(query, params)
    result = cur.fetchone()
    return result[0] if result else 0


def refresh_kpis():
    """Query DB and update KPI cards (role-aware)."""
    try:
        # Admin metrics
        total_rooms = get_count("SELECT COUNT(*) FROM user_rooms")
        available_rooms = get_count("SELECT COUNT(*) FROM user_rooms WHERE room_status = 'Available'")
        occupied = total_rooms - available_rooms
        occupancy_rate = f"{int((occupied/total_rooms)*100)}%" if total_rooms else "0%"
        # revenue from invoices
        try:
            conn2 = connect_to_database(); cur2 = conn2.cursor()
            cur2.execute("SELECT COALESCE(SUM(amount),0) FROM invoices")
            rev_row = cur2.fetchone()
            revenue = rev_row[0] if rev_row and rev_row[0] is not None else 0
            conn2.close()
        except Exception:
            revenue = 0

        # pending payments - only count unpaid invoices
        pending_payments = get_count("SELECT COUNT(*) FROM invoices WHERE status='Unpaid'")

        # Calculate average stay based on currently checked-in guests
        try:
            conn2 = connect_to_database(); cur2 = conn2.cursor()
            cur2.execute("""
                SELECT COUNT(*) FROM guest_bookings 
                WHERE status='Checked In'
            """)
            checked_in = cur2.fetchone()[0] if cur2.fetchone() else 0
            conn2.close()
            
            if checked_in > 0:
                average_stay = f"{checked_in} guests"
            else:
                average_stay = "0 guests"
        except Exception:
            average_stay = "0 guests"

        # staff/receptionist metrics (best-effort derived)
        active_bookings = get_count("SELECT COUNT(*) FROM guest_bookings WHERE status='Active'")
        today = datetime.date.today().strftime("%d-%m-%Y")
        today_checkins = get_count("SELECT COUNT(*) FROM guest_bookings WHERE check_in_date = ?", (today,))
        unread_msgs = 0
        try:
            if current_user:
                unread_msgs = get_count("SELECT COUNT(*) FROM messages WHERE receiver=? AND is_read=0", (current_user,))
        except Exception:
            unread_msgs = 0

        # update labels if present
        if isinstance(kpi_labels, dict):
            if 'Total Rooms' in kpi_labels:
                kpi_labels['Total Rooms'].config(text=str(total_rooms))
            if 'Occupancy Rate' in kpi_labels:
                kpi_labels['Occupancy Rate'].config(text=str(occupancy_rate))
            if 'Revenue' in kpi_labels:
                kpi_labels['Revenue'].config(text=f"${revenue}")
            if 'Pending Tasks' in kpi_labels:
                kpi_labels['Pending Tasks'].config(text=str(pending_payments))
            if 'Pending Payments' in kpi_labels:
                kpi_labels['Pending Payments'].config(text=str(pending_payments))

            if 'Room Requests' in kpi_labels:
                kpi_labels['Room Requests'].config(text=str(active_bookings))
            if 'Check-ins Today' in kpi_labels:
                kpi_labels['Check-ins Today'].config(text=str(today_checkins))
            if 'Messages' in kpi_labels:
                kpi_labels['Messages'].config(text=str(unread_msgs))
            if 'Average stay' in kpi_labels:
                kpi_labels['Average stay'].config(text=average_stay)
    except Exception:
        pass


def start_kpi_refresh(interval_ms=5000):
    global _kpi_refresh_job
    try:
        # cancel previous
        if _kpi_refresh_job:
            root.after_cancel(_kpi_refresh_job)
    except Exception:
        pass
    def _loop():
        refresh_kpis()
        try:
            _kpi_refresh_job = root.after(interval_ms, _loop)
        except Exception:
            pass
    _loop()

def stop_kpi_refresh():
    global _kpi_refresh_job
    try:
        if _kpi_refresh_job:
            root.after_cancel(_kpi_refresh_job)
            _kpi_refresh_job = None
    except Exception:
        pass

total_rooms = get_count("SELECT COUNT(*) FROM user_rooms")

available_rooms = get_count(
    "SELECT COUNT(*) FROM user_rooms WHERE room_status = 'Available'"
)

occupied_rooms = total_rooms - available_rooms

today = datetime.date.today().strftime("%d-%m-%Y")

today_checkins = get_count(
    "SELECT COUNT(*) FROM guest_bookings WHERE check_in_date = ?",
    (today,)
)

# Calculate revenue from paid invoices
try:
    conn2 = connect_to_database(); cur2 = conn2.cursor()
    cur2.execute("SELECT COALESCE(SUM(amount),0) FROM invoices WHERE status='Paid'")
    rev_row = cur2.fetchone()
    revenue = rev_row[0] if rev_row and rev_row[0] is not None else 0
    conn2.close()
except Exception:
    revenue = 0

# Get pending payments count
pending_payments = get_count("SELECT COUNT(*) FROM invoices WHERE status='Unpaid'")

# Calculate average stay based on currently checked-in guests
try:
    conn2 = connect_to_database(); cur2 = conn2.cursor()
    cur2.execute("""
        SELECT COUNT(*) FROM guest_bookings 
        WHERE status='Checked In'
    """)
    result = cur2.fetchone()
    checked_in = result[0] if result and result[0] else 0
    conn2.close()
    average_stay = f"{checked_in} guests" if checked_in > 0 else "0 guests"
except Exception:
    average_stay = "0 guests"

def logout():
    global current_user, current_user_role
    if confirm_action("Are you sure you want to logout?"):
        try:
            if current_user:
                set_user_offline(current_user)
        except Exception:
            pass
        # clear session
        current_user = None
        current_user_role = None
        try:
            role_label.config(text="")
            profile_img_label.config(image="")
        except Exception:
            pass
        # close all other windows and show login only
        close_all_toplevels()
        # stop polling
        try:
            stop_polling()
        except Exception:
            pass
        # hide main content
        try:
            top_bar.pack_forget()
            side_bar_container.pack_forget()
            main_content.pack_forget()
        except Exception:
            pass
        root.withdraw()
        # Show login window
        face_recognition_window()
        login_window()

# ===== GUESTS WINDOW =====
def guests_window():
    """Display and manage guests"""
    guests_win = tk.Toplevel(root)
    register_window(guests_win)
    guests_win.title("Guests Management")
    guests_win.geometry("950x650")
    guests_win.configure(bg="#f4f6f8")
    
    tk.Label(guests_win, text="Guests Management", font=("segoe", 16, "bold"), bg="#f4f6f8").pack(pady=10)
    
    # Buttons frame
    button_frame = tk.Frame(guests_win, bg="#f4f6f8")
    button_frame.pack(pady=10)
    
    def add_guest():
        add_win = tk.Toplevel(guests_win)
        register_window(add_win)
        add_win.title("Add Guest")
        add_win.geometry("400x320")
        
        tk.Label(add_win, text="Guest Name:").pack(pady=5)
        guest_name_entry = tk.Entry(add_win)
        guest_name_entry.pack(pady=5)
        
        tk.Label(add_win, text="Phone:").pack(pady=5)
        phone_entry = tk.Entry(add_win)
        phone_entry.pack(pady=5)
        
        tk.Label(add_win, text="Email:").pack(pady=5)
        email_entry = tk.Entry(add_win)
        email_entry.pack(pady=5)
        
        tk.Label(add_win, text="Address:").pack(pady=5)
        address_entry = tk.Entry(add_win)
        address_entry.pack(pady=5)
        
        def save_guest():
            try:
                guest_name = guest_name_entry.get().strip()
                phone = phone_entry.get().strip()
                email = email_entry.get().strip()
                address = address_entry.get().strip()
                
                if not guest_name:
                    show_error_message("Please enter guest name")
                    return
                
                ts = get_iso_timestamp()
                cursor.execute("INSERT INTO guests (guest_name, phone, email, address, created_at) VALUES (?, ?, ?, ?, ?)",
                              (guest_name, phone, email, address, ts))
                conn.commit()
                show_success_message("Guest added successfully")
                refresh_guests_list()
                # Clear form fields
                guest_name_entry.delete(0, tk.END)
                phone_entry.delete(0, tk.END)
                email_entry.delete(0, tk.END)
                address_entry.delete(0, tk.END)
                
            except Exception as e:
                show_error_message(f"Error: {str(e)}")
        
        tk.Button(add_win, text="Save", command=save_guest, bg="#4CAF50", fg="white").pack(pady=10)
    
    def refresh_guests_list():
        for widget in list_frame.winfo_children():
            widget.destroy()
        
        guests = get_all_guests()
        if not guests:
            tk.Label(list_frame, text="No guests found", bg="#f4f6f8").pack(pady=10)
        else:
            for guest in guests:
                guest_text = f"Name: {guest[1]} | Phone: {guest[2] or 'N/A'} | Email: {guest[3] or 'N/A'}"
                guest_frame = tk.Frame(list_frame, bg="white", relief="solid", bd=1)
                guest_frame.pack(fill="x", padx=10, pady=5)
                
                tk.Label(guest_frame, text=guest_text, bg="white", anchor="w").pack(side="left", fill="x", expand=True, padx=10, pady=5)
                
                def edit_guest_handler(g=guest):
                    edit_win = tk.Toplevel(guests_win)
                    register_window(edit_win)
                    edit_win.title("Edit Guest")
                    edit_win.geometry("400x300")
                    
                    tk.Label(edit_win, text="Guest Name:").pack(pady=5)
                    guest_name_entry = tk.Entry(edit_win)
                    guest_name_entry.insert(0, g[1])
                    guest_name_entry.pack(pady=5)
                    
                    tk.Label(edit_win, text="Phone:").pack(pady=5)
                    phone_entry = tk.Entry(edit_win)
                    phone_entry.insert(0, g[2] or "")
                    phone_entry.pack(pady=5)
                    
                    tk.Label(edit_win, text="Email:").pack(pady=5)
                    email_entry = tk.Entry(edit_win)
                    email_entry.insert(0, g[3] or "")
                    email_entry.pack(pady=5)
                    
                    tk.Label(edit_win, text="Address:").pack(pady=5)
                    address_entry = tk.Entry(edit_win)
                    address_entry.insert(0, g[4] or "")
                    address_entry.pack(pady=5)
                    
                    def save_changes():
                        try:
                            new_name = guest_name_entry.get().strip()
                            new_phone = phone_entry.get().strip()
                            new_email = email_entry.get().strip()
                            new_address = address_entry.get().strip()
                            
                            if not new_name:
                                show_error_message("Please enter guest name")
                                return
                            
                            cursor.execute("UPDATE guests SET guest_name=?, phone=?, email=?, address=? WHERE guest_id=?",
                                          (new_name, new_phone, new_email, new_address, g[0]))
                            conn.commit()
                            edit_win.destroy()
                            refresh_guests_list()
                        except Exception as e:
                            show_error_message(f"Error: {str(e)}")
                    
                    tk.Button(edit_win, text="Update", command=save_changes, bg="#2196F3", fg="white").pack(pady=10)
                
                def delete_guest_handler(g=guest):
                    if confirm_action(f"Delete guest {g[1]}?"):
                        delete_guest(g[1])
                        refresh_guests_list()
                
                tk.Button(guest_frame, text="Edit", command=edit_guest_handler, bg="#2196F3", fg="white", width=5).pack(side="right", padx=5, pady=5)
                tk.Button(guest_frame, text="Delete", command=delete_guest_handler, bg="#f44336", fg="white", width=7).pack(side="right", padx=5, pady=5)
    
    tk.Button(button_frame, text="Add New Guest", command=add_guest, bg="#4CAF50", fg="white").pack(side="left", padx=5)
    
    # Scrollable list frame
    canvas = tk.Canvas(guests_win, bg="#f4f6f8", highlightthickness=0)
    scrollbar = tk.Scrollbar(guests_win, orient="vertical", command=canvas.yview)
    list_frame = tk.Frame(canvas, bg="#f4f6f8")
    list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    canvas.create_window((0, 0), window=list_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True, padx=10)
    scrollbar.pack(side="right", fill="y")

# ===== ACTIVE MENU HIGHLIGHT =====
    


# ===== ACTIVE MENU HIGHLIGHT =====
ACTIVE_BG = "#c9a24d"
DEFAULT_BG = "#162030"

def set_active_menu(name):
    for btn in menu_buttons.values():
        btn.config(ACTIVE_BG, fg="white")
        menu_buttons[name].config(bg=ACTIVE_BG,fg="black")

def fetch_one(query, params=()):
    conn = connect_to_database()
    cur = conn.cursor()
    cur.execute(query, params)
    result = cur.fetchone()
    return result[0] if result else 0

def start_login():
    global root
    try:
        # if root exists use it, otherwise create a hidden root for login
        if 'root' not in globals() or root is None:
            root = tk.Tk()
            root.withdraw()
        else:
            root.withdraw()
    except Exception:
        root = tk.Tk()
        root.withdraw()
    face_recognition_window()


### Minimal implementations for menu windows (so 'not implemented' items work)
def dashboard_window():
    dw = tk.Toplevel(root)
    register_window(dw)
    dw.title("Dashboard")
    dw.geometry("1100x600")
    dw.configure(bg=COLOR_DARK_BG)
    
    # Title
    tk.Label(dw, text="Dashboard Overview", font=("segoe ui", 18, "bold"), 
             bg=COLOR_DARK_BG, fg=COLOR_GOLD).pack(pady=20)
    
    # Role-specific cards frame
    cards_container = tk.Frame(dw, bg=COLOR_DARK_BG)
    cards_container.pack(fill="both", expand=True, padx=20, pady=10)
    
    # Define role-specific KPIs
    role_kpis = {
        "admin": [("Total Rooms", "🏨"), ("Occupancy Rate", "📊"), ("Revenue", "💰"), ("Pending Tasks", "📋")],
        "staff": [("Your Tasks", "✅"), ("Room Requests", "🔔"), ("Maintenance Items", "🔧"), ("Messages", "💬")],
        "receptionist": [("Check-ins Today", "📅"), ("Pending Bookings", "📝"), ("Guest Inquiries", "❓"), ("Messages", "💬")]
    }
    
    kpis = role_kpis.get(current_user_role, role_kpis["staff"])
    
    # Create KPI cards in grid
    global kpi_labels
    kpi_labels = {}

    # mapping of KPI to action name (function name string)
    kpi_action_map = {
        "Total Rooms": "rooms_window",
        "Occupancy Rate": "bookings_window",
        "Revenue": "invoices_window",
        "Pending Tasks": "maintenance_window",
        "Your Tasks": "housekeeping_window",
        "Room Requests": "bookings_window",
        "Maintenance Items": "maintenance_window",
        "Check-ins Today": "bookings_window",
        "Pending Bookings": "bookings_window",
        "Guest Inquiries": "guests_window",
    }

    def open_kpi(name):
        fn_name = kpi_action_map.get(name)
        if not fn_name:
            return
        cmd = globals().get(fn_name)
        if callable(cmd):
            # role-based safeguard: only open if user is allowed for that menu
            try:
                # find corresponding menu button and check allowed_roles
                mb = menu_buttons.get(cmd.__name__.replace('_window','').replace('window','').title())
            except Exception:
                mb = None
            cmd()

    for idx, (label, icon) in enumerate(kpis):
        col = idx % 2
        row = idx // 2
        card = tk.Frame(cards_container, bg=COLOR_SIDEBAR, relief="flat", bd=1)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        tk.Label(card, text=icon, font=("Arial", 24), bg=COLOR_SIDEBAR).pack(pady=10)
        tk.Label(card, text=label, font=("segoe ui", 12, "bold"), 
                bg=COLOR_SIDEBAR, fg="white").pack()
        value_label = tk.Label(card, text="--", font=("segoe ui", 20, "bold"), 
                bg=COLOR_SIDEBAR, fg=COLOR_GOLD)
        value_label.pack(pady=5)
        # store reference so refresh_kpis can update
        kpi_labels[label] = value_label
        # bind click on card and children to open related window
        card.bind("<Button-1>", lambda e, n=label: open_kpi(n))
        for child in card.winfo_children():
            child.bind("<Button-1>", lambda e, n=label: open_kpi(n))
    
    cards_container.columnconfigure(0, weight=1)
    cards_container.columnconfigure(1, weight=1)
    # populate initial KPI values
    try:
        refresh_kpis()
    except Exception:
        pass

def rooms_window():
    rw = tk.Toplevel(root)
    register_window(rw)
    rw.title("Manage Rooms")
    rw.geometry("750x500")

    # Treeview for rooms
    cols = ("room_number", "room_type", "room_price", "room_status")
    tree = ttk.Treeview(rw, columns=cols, show='headings')
    for c in cols:
        tree.heading(c, text=c.replace('_', ' ').title())
    tree.pack(fill='both', expand=True, padx=10, pady=10)

    def load_rooms():
        for r in tree.get_children():
            tree.delete(r)
        cursor.execute("SELECT room_number, room_type, room_price, room_status FROM user_rooms ORDER BY room_number")
        for row in cursor.fetchall():
            tree.insert('', 'end', values=row)

    def add_room():
        aw = tk.Toplevel(rw)
        register_window(aw)
        aw.title('Add Room')
        tk.Label(aw, text='Room Number').pack()
        rn = tk.Entry(aw); rn.pack()
        tk.Label(aw, text='Type').pack()
        rt = ttk.Combobox(aw, values=['Single','Double','Suite'], state='readonly'); rt.pack()
        tk.Label(aw, text='Price').pack()
        rp = tk.Entry(aw); rp.pack()
        tk.Label(aw, text='Status').pack()
        rs = ttk.Combobox(aw, values=['Available','Occupied','Maintenance'], state='readonly'); rs.pack()
        def save():
            try:
                cursor.execute("INSERT INTO user_rooms (room_number, room_type, room_price, room_status) VALUES (?, ?, ?, ?)", (int(rn.get()), rt.get(), float(rp.get()), rs.get()))
                conn.commit()
                show_success_message("Room added successfully")
                load_rooms()
                # Clear form fields
                rn.delete(0, tk.END)
                rt.set('')
                rp.delete(0, tk.END)
                rs.set('')
            except Exception as e:
                show_error_message(str(e))
        tk.Button(aw, text='Save', command=save).pack(pady=10)

    def edit_room():
        sel = tree.selection()
        if not sel: return
        item = tree.item(sel[0])['values']
        ew = tk.Toplevel(rw); register_window(ew); ew.title('Edit Room')
        tk.Label(ew, text='Room Number').pack(); rn = tk.Entry(ew); rn.insert(0,str(item[0])); rn.pack()
        tk.Label(ew, text='Type').pack(); rt = ttk.Combobox(ew, values=['Single','Double','Suite'], state='readonly'); rt.set(item[1]); rt.pack()
        tk.Label(ew, text='Price').pack(); rp = tk.Entry(ew); rp.insert(0,str(item[2])); rp.pack()
        tk.Label(ew, text='Status').pack(); rs = ttk.Combobox(ew, values=['Available','Occupied','Maintenance'], state='readonly'); rs.set(item[3]); rs.pack()
        def save():
            try:
                cursor.execute("UPDATE user_rooms SET room_type=?, room_price=?, room_status=? WHERE room_number=?", (rt.get(), float(rp.get()), rs.get(), int(rn.get())))
                conn.commit()
                show_success_message("Room updated successfully")
                load_rooms()
            except Exception as e:
                show_error_message(str(e))
        tk.Button(ew, text='Update', command=save).pack(pady=10)

    def delete_room():
        sel = tree.selection()
        if not sel: return
        item = tree.item(sel[0])['values']
        if confirm_action(f"Delete room {item[0]}?"):
            try:
                cursor.execute("DELETE FROM user_rooms WHERE room_number=?", (item[0],))
                conn.commit(); load_rooms()
            except Exception as e:
                show_error_message(str(e))

    btnf = tk.Frame(rw); btnf.pack(fill='x')
    tk.Button(btnf, text='Add', command=add_room, bg='#4CAF50', fg='white').pack(side='left', padx=5, pady=5)
    tk.Button(btnf, text='Edit', command=edit_room, bg='#2196F3', fg='white').pack(side='left', padx=5, pady=5)
    tk.Button(btnf, text='Delete', command=delete_room, bg='#f44336', fg='white').pack(side='left', padx=5, pady=5)

    load_rooms()

def bookings_window():
    bw = tk.Toplevel(root)
    register_window(bw)
    bw.title("Manage Bookings")
    bw.geometry("950x600")

    cols = ("booking_id","guest_name","room_number","check_in","check_out","status","total_amount")
    tree = ttk.Treeview(bw, columns=cols, show='headings')
    for c in cols:
        tree.heading(c, text=c.replace('_', ' ').title())
    tree.pack(fill='both', expand=True, padx=10, pady=10)

    def load_bookings():
        for r in tree.get_children(): tree.delete(r)
        cursor.execute("SELECT booking_id, guest_name, room_number, check_in_date, check_out_date, status, total_amount FROM guest_bookings ORDER BY booking_id DESC")
        for row in cursor.fetchall():
            tree.insert('', 'end', values=row)

    def add_booking():
        aw = tk.Toplevel(bw); register_window(aw); aw.title('New Booking')
        tk.Label(aw, text='Guest Name').pack(); gn = tk.Entry(aw); gn.pack()
        tk.Label(aw, text='Room Number').pack(); rn = ttk.Combobox(aw, values=[r[0] for r in cursor.execute('SELECT room_number FROM user_rooms').fetchall()], state='readonly'); rn.pack()
        tk.Label(aw, text='Check-in (DD-MM-YYYY)').pack(); ci = tk.Entry(aw); ci.pack()
        tk.Label(aw, text='Check-out (DD-MM-YYYY)').pack(); co = tk.Entry(aw); co.pack()
        def save():
            try:
                guest = gn.get().strip(); room = int(rn.get()); cind = ci.get().strip(); cout = co.get().strip()
                # overlap prevention
                cursor.execute("SELECT check_in_date, check_out_date FROM guest_bookings WHERE room_number=? AND status!='Checked Out'", (room,))
                for ex in cursor.fetchall():
                    d1 = datetime.datetime.strptime(ex[0], "%d-%m-%Y")
                    d2 = datetime.datetime.strptime(ex[1], "%d-%m-%Y")
                    new1 = datetime.datetime.strptime(cind, "%d-%m-%Y")
                    new2 = datetime.datetime.strptime(cout, "%d-%m-%Y")
                    if not (new2 <= d1 or new1 >= d2):
                        show_error_message('Room already booked for selected dates')
                        return
                # get price
                cursor.execute('SELECT room_price FROM user_rooms WHERE room_number=?', (room,))
                rp = cursor.fetchone()
                price = rp[0] if rp else 0
                total = calculate_total_amount(price, cind, cout)
                # generate UUID for booking_id
                booking_id = generate_booking_id()
                cursor.execute("INSERT INTO guest_bookings (booking_id, guest_name, room_number, check_in_date, check_out_date, status, total_amount) VALUES (?, ?, ?, ?, ?, ?, ?)", (booking_id, guest, room, cind, cout, 'Active', total))
                conn.commit()
                notify_new_booking(guest, room, cind, cout)
                show_success_message(f"Booking created for {guest}")
                load_bookings()
                refresh_kpis()
                # Clear form fields
                gn.delete(0, tk.END)
                rn.set('')
                ci.delete(0, tk.END)
                co.delete(0, tk.END)
            except Exception as e:
                show_error_message(str(e))
        tk.Button(aw, text='Save', command=save, bg='#4CAF50', fg='white').pack(pady=10)


    def check_in():
        sel = tree.selection()
        if not sel: return
        item = tree.item(sel[0])['values']
        booking_id = item[0]
        cursor.execute("UPDATE guest_bookings SET status='Checked In' WHERE booking_id=?", (booking_id,))
        cursor.execute("UPDATE user_rooms SET room_status='Occupied' WHERE room_number=?", (item[2],))
        conn.commit(); load_bookings(); refresh_kpis()

    def check_out():
        sel = tree.selection()
        if not sel: return
        item = tree.item(sel[0])['values']
        booking_id = item[0]
        guest_name = item[1]
        room_number = item[2]
        cursor.execute("UPDATE guest_bookings SET status='Checked Out' WHERE booking_id=?", (booking_id,))
        cursor.execute("UPDATE user_rooms SET room_status='Needs Cleaning' WHERE room_number=?", (room_number,))
        
        # Create housekeeping task for room cleaning
        ts = get_iso_timestamp()
        cursor.execute("""
            INSERT INTO housekeeping_tasks (room_number, task_type, status, created_at)
            VALUES (?, ?, ?, ?)
        """, (room_number, 'Room Cleaning', 'Pending', ts))
        
        conn.commit()
        notify_checkout(guest_name, room_number)
        # Notify housekeeping staff
        notify_housekeeping_task("Room Cleaning", room_number)
        load_bookings()
        refresh_kpis()
        
    btnf = tk.Frame(bw); btnf.pack(fill='x', padx=10, pady=10)
    tk.Button(btnf, text='New Booking', command=add_booking, bg='#4CAF50', fg='white').pack(side='left', padx=5, pady=5)
    tk.Button(btnf, text='Check In', command=check_in, bg='#2196F3', fg='white').pack(side='left', padx=5, pady=5)
    tk.Button(btnf, text='Check Out', command=check_out, bg='#f57c00', fg='white').pack(side='left', padx=5, pady=5)
    
    load_bookings()

def invoices_window():
    iw = tk.Toplevel(root)
    register_window(iw)
    iw.title("💳 Invoices & Payments")
    iw.geometry("850x600")

    cols = ("id","booking_id","guest_name","amount","status","created_at")
    tree = ttk.Treeview(iw, columns=cols, show='headings')
    for c in cols:
        tree.heading(c, text=c.replace('_', ' ').title())
    tree.pack(fill='both', expand=True, padx=10, pady=10)


    def load_invoices():
        for r in tree.get_children(): tree.delete(r)
        cursor.execute("SELECT id, booking_id, guest_name, amount, status, created_at FROM invoices ORDER BY id DESC")
        for row in cursor.fetchall(): tree.insert('', 'end', values=row)
        
    def generate_invoice_id():
        invoice = str(random.randint(100000,999999))
        cursor.execute('SELECT id FROM invoices WHERE id=?', (invoice,))
        if cursor.fetchone():
            return generate_invoice_id()
        return invoice

    def create_invoice():
        ci = tk.Toplevel(iw); register_window(ci); ci.title('Create Invoice'); ci.geometry("400x250")
        tk.Label(ci, text='Booking ID').pack(); bid = tk.Entry(ci); bid.pack()
        def save():
            try:
                b = bid.get().strip()
                cursor.execute('SELECT guest_name, total_amount FROM guest_bookings WHERE booking_id=?', (b,))
                r = cursor.fetchone()
                if not r: show_error_message('Booking not found'); return
                guest, amt = r
                ts = get_iso_timestamp()
                inv_id = generate_invoice_id()
                cursor.execute('INSERT INTO invoices (id, booking_id, guest_name, amount, status, created_at) VALUES (?, ?, ?, ?, ?, ?)', (inv_id, b, guest, amt, 'Unpaid', ts))
                conn.commit()
                notify_invoice_created(b, amt)
                show_success_message(f'Invoice #{inv_id} created successfully')
                load_invoices()
                # Clear form
                bid.delete(0, tk.END)
            except ValueError:
                show_error_message('Invalid booking ID format')
            except Exception as e:
                show_error_message(str(e))
        tk.Button(ci, text='Create', command=save, bg='#4CAF50', fg='white').pack(pady=10)

    def mark_paid():
        sel = tree.selection()
        if not sel:
            show_error_message("Please select an invoice to mark as paid")
            return
        item = tree.item(sel[0])['values']
        invoice_id = item[0]
        status = item[4]
        if status == "Paid":
            show_error_message(f"Invoice #{invoice_id} is already marked as Paid")
            return
        if not confirm_action(f"Mark invoice #{invoice_id} as paid?"):
            return
        cursor.execute("UPDATE invoices SET status='Paid' WHERE id=?", (invoice_id,))
        conn.commit()
        tree.item(sel[0], values=(item[0], item[1], item[2], item[3], "Paid", item[5]))
        create_notification(current_user, f'✅ Invoice #{invoice_id} marked as Paid')
        show_success_message(f'Invoice #{invoice_id} marked as Paid')

    btnf = tk.Frame(iw); btnf.pack(fill='x', padx=10, pady=10)
    tk.Button(btnf, text='Create Invoice', command=create_invoice, bg='#4CAF50', fg='white').pack(side='left', padx=5, pady=5)
    tk.Button(btnf, text='Mark Paid', command=mark_paid, bg='#2196F3', fg='white').pack(side='left', padx=5, pady=5)
    tk.Button(btnf, text='Export PDF', command=lambda: export_selected_invoice(tree), bg='#6a1b9a', fg='white').pack(side='left', padx=5, pady=5)

    load_invoices()

def export_selected_invoice(tree):
    sel = tree.selection()
    if not sel:
        show_error_message('Select an invoice to export')
        return
    item = tree.item(sel[0])['values']
    invoice_id = item[0]
    export_invoice_pdf(invoice_id)

def export_invoice_pdf(invoice_id):
    # ensure folder
    outdir = os.path.join(os.getcwd(), 'invoices_pdf')
    os.makedirs(outdir, exist_ok=True)
    cursor.execute('SELECT id, booking_id, guest_name, amount, status, created_at FROM invoices WHERE id=?', (invoice_id,))
    inv = cursor.fetchone()
    if not inv:
        show_error_message('Invoice not found')
        return
    filename = os.path.join(outdir, f'invoice_{invoice_id}.pdf')
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm

        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4

        # Header: logo if available + hotel name
        y = height - 30*mm
        try:
            if 'logo_image' in globals() and logo_image:
                logo_path = 'temp_logo_invoice.png'
                try:
                    logo_image.resize((100,100)).save(logo_path)
                    c.drawImage(logo_path, 20*mm, y - 20*mm, width=30*mm, height=30*mm)
                    try: os.remove(logo_path)
                    except Exception: pass
                except Exception:
                    pass
        except Exception:
            pass

        c.setFont('Helvetica-Bold', 18)
        c.drawCentredString(width/2, y, 'GRAND SUIT HOTELS - Invoice')

        # Invoice metadata
        c.setFont('Helvetica', 11)
        left_x = 20*mm
        meta_y = y - 12*mm
        c.drawString(left_x, meta_y, f'Invoice #: {inv[0]}')
        c.drawString(left_x + 80*mm, meta_y, f'Date: {inv[5]}')
        meta_y -= 7*mm
        c.drawString(left_x, meta_y, f'Guest: {inv[2]}')
        c.drawString(left_x + 80*mm, meta_y, f'Booking ID: {inv[1]}')

        # Booking details (try to fetch booking info)
        try:
            cursor.execute('SELECT room_number, check_in_date, check_out_date, total_amount FROM guest_bookings WHERE booking_id=?', (inv[1],))
            binfo = cursor.fetchone()
        except Exception:
            binfo = None

        table_y = meta_y - 14*mm
        c.setFont('Helvetica-Bold', 12)
        c.drawString(left_x, table_y, 'Description')
        c.drawRightString(width - 20*mm, table_y, 'Amount')
        c.setLineWidth(0.5)
        c.line(left_x, table_y - 2*mm, width - 20*mm, table_y - 2*mm)

        c.setFont('Helvetica', 11)
        table_y -= 8*mm
        desc = 'Room charges'
        amount = inv[3]
        if binfo:
            desc = f'Room {binfo[0]} ({binfo[1]} to {binfo[2]})'
            try:
                amount = binfo[3]
            except Exception:
                amount = inv[3]
        c.drawString(left_x, table_y, desc)
        c.drawRightString(width - 20*mm, table_y, f"₦{amount:,.2f}")

        # Totals
        table_y -= 12*mm
        c.setFont('Helvetica-Bold', 12)
        c.drawString(left_x, table_y, 'Total')
        c.drawRightString(width - 20*mm, table_y, f"₦{amount:,.2f}")

        # Status note
        table_y -= 10*mm
        c.setFont('Helvetica', 10)
        c.drawString(left_x, table_y, f'Status: {inv[4]}')

        # Footer
        c.setFont('Helvetica-Oblique', 9)
        c.drawCentredString(width/2, 15*mm, 'Thank you for staying with GRAND SUIT HOTELS')

        c.showPage()
        c.save()
        show_success_message(f'PDF exported: {filename}')
    except Exception:
        # fallback text export
        try:
            txt = os.path.join(outdir, f'invoice_{invoice_id}.txt')
            with open(txt, 'w', encoding='utf-8') as f:
                f.write(f'Invoice #{inv[0]}\nGuest: {inv[2]}\nBooking ID: {inv[1]}\nAmount: {inv[3]}\nStatus: {inv[4]}\nDate: {inv[5]}\n')
            show_success_message(f'Exported text invoice: {txt}')
        except Exception as e:
            show_error_message(f'Failed to export invoice: {e}')

def reports_window():
    rw = tk.Toplevel(root)
    register_window(rw)
    rw.title("Reports & Analytics")
    rw.geometry("900x600")
    rw.configure(bg="#f4f6f8")
    
    tk.Label(rw, text="Reports & Analytics", font=("segoe ui", 18, "bold"), bg="#f4f6f8", fg="#1a1a1a").pack(pady=20)
    
    # Create tabs for different reports
    tabs_frame = tk.Frame(rw, bg="#f4f6f8")
    tabs_frame.pack(fill="x", padx=20, pady=10)
    
    report_frames = {}
    
    def show_report(report_name):
        for f in report_frames.values():
            f.pack_forget()
        report_frames[report_name].pack(fill="both", expand=True, padx=20, pady=10)
    
    # Occupancy Report Tab
    occupancy_frame = tk.Frame(rw, bg="#ffffff", relief="solid", bd=1)
    report_frames["Occupancy"] = occupancy_frame
    tk.Label(occupancy_frame, text="Occupancy Report", font=("segoe ui", 14, "bold"), bg="#ffffff").pack(pady=10)
    try:
        total = get_count("SELECT COUNT(*) FROM user_rooms")
        occupied = get_count("SELECT COUNT(*) FROM user_rooms WHERE room_status='Occupied'")
        rate = int((occupied/total)*100) if total > 0 else 0
        tk.Label(occupancy_frame, text=f"Total Rooms: {total}\nOccupied: {occupied}\nOccupancy Rate: {rate}%", 
                 font=("segoe ui", 12), bg="#ffffff").pack(pady=20)
    except Exception as e:
        tk.Label(occupancy_frame, text=f"Error loading report: {e}", font=("segoe ui", 10), bg="#ffffff", fg="red").pack(pady=20)
    
    # Revenue Report Tab
    revenue_frame = tk.Frame(rw, bg="#ffffff", relief="solid", bd=1)
    report_frames["Revenue"] = revenue_frame
    tk.Label(revenue_frame, text="Revenue Report", font=("segoe ui", 14, "bold"), bg="#ffffff").pack(pady=10)
    try:
        conn2 = connect_to_database()
        cur2 = conn2.cursor()
        cur2.execute("SELECT COALESCE(SUM(amount),0) FROM invoices WHERE status='Paid'")
        paid = cur2.fetchone()[0] if cur2.fetchone() else 0
        cur2.execute("SELECT COALESCE(SUM(amount),0) FROM invoices WHERE status!='Paid'")
        pending = cur2.fetchone()[0] if cur2.fetchone() else 0
        conn2.close()
        tk.Label(revenue_frame, text=f"Total Revenue (Paid): ${paid}\nPending: ${pending}", 
                 font=("segoe ui", 12), bg="#ffffff").pack(pady=20)
    except Exception as e:
        tk.Label(revenue_frame, text=f"Error loading report: {e}", font=("segoe ui", 10), bg="#ffffff", fg="red").pack(pady=20)
    
    # Buttons to switch reports
    tk.Button(tabs_frame, text="Occupancy", command=lambda: show_report("Occupancy"), bg="#4CAF50", fg="white").pack(side="left", padx=5)
    tk.Button(tabs_frame, text="Revenue", command=lambda: show_report("Revenue"), bg="#2196F3", fg="white").pack(side="left", padx=5)
    
    # Show default report
    show_report("Occupancy")

def housekeeping_window():
    hw = tk.Toplevel(root)
    register_window(hw)
    hw.title("Housekeeping - Rooms Needing Cleaning")
    hw.geometry("900x600")
    hw.configure(bg="#f4f6f8")
    
    tk.Label(hw, text="Rooms Needing Cleaning", font=("segoe ui", 16, "bold"), bg="#f4f6f8").pack(pady=10)
    
    # Treeview for rooms needing cleaning
    cols = ("room_number", "room_type", "guest_name", "check_out_date", "task_status", "created_at")
    tree = ttk.Treeview(hw, columns=cols, show='headings', height=15)
    for c in cols:
        tree.heading(c, text=c.replace('_', ' ').title())
        tree.column(c, width=120)
    tree.pack(fill='both', expand=True, padx=10, pady=10)
    
    def load_cleaning_tasks():
        for r in tree.get_children():
            tree.delete(r)
        try:
            # Get rooms that just had checkout
            cursor.execute("""
                SELECT ht.room_number, ur.room_type, gb.guest_name, gb.check_out_date, ht.status, ht.created_at
                FROM housekeeping_tasks ht
                JOIN user_rooms ur ON ht.room_number = ur.room_number
                LEFT JOIN guest_bookings gb ON ur.room_number = gb.room_number AND gb.status = 'Checked Out'
                WHERE ht.status != 'Completed'
                ORDER BY ht.created_at DESC
            """)
            for row in cursor.fetchall():
                tree.insert('', 'end', values=row)
        except Exception as e:
            show_error_message(f"Error loading tasks: {str(e)}")
    
    def mark_completed():
        sel = tree.selection()
        if not sel:
            show_error_message("Please select a room to mark as cleaned")
            return
        item = tree.item(sel[0])['values']
        room_number = item[0]
        try:
            ts = get_iso_timestamp()
            cursor.execute("UPDATE housekeeping_tasks SET status='Completed', completed_at=? WHERE room_number=? AND status != 'Completed'", (ts, room_number))
            cursor.execute("UPDATE user_rooms SET room_status='Available' WHERE room_number=?", (room_number,))
            conn.commit()
            create_notification(current_user, f"🧹 Room {room_number} cleaning completed")
            load_cleaning_tasks()
        except Exception as e:
            show_error_message(f"Error: {str(e)}")
    
    def add_notes():
        sel = tree.selection()
        if not sel:
            show_error_message("Please select a room")
            return
        room_number = tree.item(sel[0])['values'][0]
        
        nw = tk.Toplevel(hw)
        register_window(nw)
        nw.title("Add Notes")
        nw.geometry("400x300")
        
        tk.Label(nw, text=f"Notes for Room {room_number}:", font=("segoe ui", 11, "bold")).pack(pady=10)
        notes_text = tk.Text(nw, width=45, height=10)
        notes_text.pack(padx=10, pady=10)
        
        def save_notes():
            notes = notes_text.get("1.0", tk.END).strip()
            try:
                cursor.execute("UPDATE housekeeping_tasks SET notes=? WHERE room_number=? AND status != 'Completed'", (notes, room_number))
                conn.commit()
                nw.destroy()
                load_cleaning_tasks()
                create_notification(current_user, f"📝 Notes added for Room {room_number}")
            except Exception as e:
                show_error_message(f"Error: {str(e)}")
        
        tk.Button(nw, text="Save Notes", command=save_notes, bg="#4CAF50", fg="white").pack(pady=10)
    
    btnf = tk.Frame(hw, bg="#f4f6f8")
    btnf.pack(fill='x', padx=10, pady=10)
    tk.Button(btnf, text="Mark Cleaned", command=mark_completed, bg="#4CAF50", fg="white").pack(side='left', padx=5)
    tk.Button(btnf, text="Add Notes", command=add_notes, bg="#2196F3", fg="white").pack(side='left', padx=5)
    tk.Button(btnf, text="Refresh", command=load_cleaning_tasks, bg="#FF9800", fg="white").pack(side='left', padx=5)
    
    load_cleaning_tasks()

def maintenance_window():
    mw = tk.Toplevel(root)
    register_window(mw)
    mw.title("Maintenance Issues")
    mw.geometry("1000x600")
    mw.configure(bg="#f4f6f8")
    
    tk.Label(mw, text="Maintenance Issues & Complaints", font=("segoe ui", 16, "bold"), bg="#f4f6f8").pack(pady=10)
    
    # Treeview for maintenance issues
    cols = ("issue_id", "room_number", "issue_type", "reported_by", "status", "created_at")
    tree = ttk.Treeview(mw, columns=cols, show='headings', height=15)
    for c in cols:
        tree.heading(c, text=c.replace('_', ' ').title())
        tree.column(c, width=130)
    tree.pack(fill='both', expand=True, padx=10, pady=10)
    
    def load_issues():
        for r in tree.get_children():
            tree.delete(r)
        try:
            cursor.execute("""
                SELECT issue_id, room_number, issue_type, reported_by, status, created_at
                FROM maintenance_issues
                WHERE status != 'Fixed'
                ORDER BY created_at DESC
            """)
            for row in cursor.fetchall():
                tree.insert('', 'end', values=row)
        except Exception as e:
            show_error_message(f"Error loading issues: {str(e)}")
    
    def view_issue():
        sel = tree.selection()
        if not sel:
            show_error_message("Please select an issue")
            return
        item = tree.item(sel[0])['values']
        issue_id = item[0]
        
        try:
            cursor.execute("SELECT * FROM maintenance_issues WHERE issue_id=?", (issue_id,))
            issue = cursor.fetchone()
            if not issue:
                return
            
            vw = tk.Toplevel(mw)
            register_window(vw)
            vw.title(f"Issue #{issue_id}")
            vw.geometry("500x400")
            
            info_text = f"""Issue ID: {issue[0]}
Room: {issue[1] if issue[1] else 'General'}
Type: {issue[2]}
Description: {issue[3]}
Status: {issue[4]}
Reported By: {issue[5]}
Created: {issue[6]}
Resolved: {issue[7] if issue[7] else 'Not resolved yet'}"""
            
            tk.Label(vw, text=info_text, font=("segoe ui", 10), justify="left").pack(anchor="w", padx=10, pady=10)
        except Exception as e:
            show_error_message(f"Error: {str(e)}")
    
    def mark_fixed():
        sel = tree.selection()
        if not sel:
            show_error_message("Please select an issue to mark as fixed")
            return
        item = tree.item(sel[0])['values']
        issue_id = item[0]
        
        try:
            ts = get_iso_timestamp()
            cursor.execute("UPDATE maintenance_issues SET status='Fixed', resolved_at=? WHERE issue_id=?", (ts, issue_id))
            
            # Get issue details for notification
            cursor.execute("SELECT reported_by, room_number, issue_type FROM maintenance_issues WHERE issue_id=?", (issue_id,))
            issue_info = cursor.fetchone()
            
            conn.commit()
            
            # Notify the person who reported the issue
            if issue_info and issue_info[0]:
                create_notification(issue_info[0], f"✅ Maintenance Issue #{issue_id} has been fixed - Room {issue_info[1] if issue_info[1] else 'General'}")
            
            # Notify all staff
            cursor.execute("SELECT username FROM users WHERE role='staff'")
            for user in cursor.fetchall():
                create_notification(user[0], f"✅ Maintenance Issue #{issue_id} - {issue_info[2] if issue_info else 'Unknown'} is now Fixed")
            
            load_issues()
        except Exception as e:
            show_error_message(f"Error: {str(e)}")
    
    btnf = tk.Frame(mw, bg="#f4f6f8")
    btnf.pack(fill='x', padx=10, pady=10)
    tk.Button(btnf, text="View Details", command=view_issue, bg="#2196F3", fg="white").pack(side='left', padx=5)
    tk.Button(btnf, text="Mark as Fixed", command=mark_fixed, bg="#4CAF50", fg="white").pack(side='left', padx=5)
    tk.Button(btnf, text="Report Issue", command=lambda: report_maintenance_issue(load_issues), bg="#FF9800", fg="white").pack(side='left', padx=5)
    
    load_issues()

def report_maintenance_issue(callback=None):
    """Allow customers or staff to report maintenance issues"""
    rw = tk.Toplevel(root)
    register_window(rw)
    rw.title("Report Maintenance Issue")
    rw.geometry("500x500")
    rw.configure(bg="#f4f6f8")
    
    tk.Label(rw, text="Report Maintenance Issue", font=("segoe ui", 14, "bold"), bg="#f4f6f8").pack(pady=10)
    
    tk.Label(rw, text="Room Number (optional):", bg="#f4f6f8").pack(anchor="w", padx=20, pady=(10, 0))
    room_entry = tk.Entry(rw)
    room_entry.pack(padx=20, pady=5, fill="x")
    
    tk.Label(rw, text="Issue Type:", bg="#f4f6f8").pack(anchor="w", padx=20, pady=(10, 0))
    issue_type = ttk.Combobox(rw, values=["Electrical", "Plumbing", "HVAC", "Furniture", "Appliance", "Other"], state="readonly")
    issue_type.pack(padx=20, pady=5, fill="x")
    
    tk.Label(rw, text="Description:", bg="#f4f6f8").pack(anchor="w", padx=20, pady=(10, 0))
    desc_text = tk.Text(rw, height=8, width=50)
    desc_text.pack(padx=20, pady=5, fill="both", expand=True)
    
    def submit_issue():
        try:
            room = room_entry.get().strip() if room_entry.get().strip() else None
            issue_t = issue_type.get()
            desc = desc_text.get("1.0", tk.END).strip()
            
            if not issue_t or not desc:
                show_error_message("Please fill all required fields")
                return
            
            ts = get_iso_timestamp()
            cursor.execute("""
                INSERT INTO maintenance_issues 
                (room_number, issue_type, description, status, reported_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (room, issue_t, desc, "Pending", current_user, ts))
            conn.commit()
            
            # Notify staff
            cursor.execute("SELECT username FROM users WHERE role='staff'")
            for user in cursor.fetchall():
                create_notification(user[0], f"🔧 New Maintenance Issue: {issue_t} - Room {room if room else 'General'}")
            
            # Notify admin
            cursor.execute("SELECT username FROM users WHERE role='admin'")
            for user in cursor.fetchall():
                create_notification(user[0], f"🔧 New Maintenance Issue reported by {current_user}")
            
            show_success_message("Issue reported successfully")
            # Clear form
            room.set('')
            issue_type.set('')
            desc_text.delete("1.0", tk.END)
            if callback:
                callback()
        except Exception as e:
            show_error_message(f"Error: {str(e)}")
    
    tk.Button(rw, text="Submit Report", command=submit_issue, bg="#4CAF50", fg="white", font=("segoe ui", 11)).pack(pady=10)

def restaurant_window():
    rw = tk.Toplevel(root)
    register_window(rw)
    rw.title("Restaurant / Room Service Orders")
    rw.geometry("950x600")
    rw.configure(bg="#f4f6f8")
    
    tk.Label(rw, text="Restaurant & Room Service Orders", font=("segoe ui", 16, "bold"), bg="#f4f6f8").pack(pady=10)
    
    # Orders table
    cols = ("order_id", "room", "items", "status", "total", "date")
    tree = ttk.Treeview(rw, columns=cols, show='headings', height=15)
    for c in cols:
        tree.heading(c, text=c.replace('_', ' ').title())
        if c == "order_id":
            tree.column(c, width=80)
        elif c == "date":
            tree.column(c, width=120)
        else:
            tree.column(c, width=120)
    tree.pack(fill='both', expand=True, padx=10, pady=10)
    
    def load_orders():
        for item in tree.get_children():
            tree.delete(item)
        try:
            cursor.execute("""SELECT order_id, room_number, items, status, total_price, order_date 
                           FROM restaurant_orders ORDER BY order_date DESC""")
            for row in cursor.fetchall():
                tree.insert('', 'end', values=row)
        except Exception:
            # Table might not exist yet
            pass
    
    load_orders()
    
    def new_order():
        nw = tk.Toplevel(rw)
        register_window(nw)
        nw.title("New Order")
        nw.geometry("400x450")
        nw.configure(bg="#E6E1E1")
        
        tk.Label(nw, text="Create New Order", font=("segoe ui", 14, "bold"), bg="#E6E1E1").pack(pady=10)
        
        tk.Label(nw, text="Room Number:", bg="#E6E1E1", font=("segoe ui", 10)).pack(pady=5)
        room_entry = tk.Entry(nw, bg="#ffffff", font=("segoe ui", 10))
        room_entry.pack(ipadx=20, pady=5, fill="x")
        
        tk.Label(nw, text="Items Ordered:", bg="#E6E1E1", font=("segoe ui", 10)).pack(pady=5)
        items_entry = tk.Entry(nw, bg="#ffffff", font=("segoe ui", 10))
        items_entry.pack(ipadx=20, pady=5, fill="x")
        
        tk.Label(nw, text="Total Price:", bg="#E6E1E1", font=("segoe ui", 10)).pack(pady=5)
        price_entry = tk.Entry(nw, bg="#ffffff", font=("segoe ui", 10))
        price_entry.pack(ipadx=20, pady=5, fill="x")
        
        tk.Label(nw, text="Status:", bg="#E6E1E1", font=("segoe ui", 10)).pack(pady=5)
        status_var = tk.StringVar(value="Pending")
        status_menu = ttk.Combobox(nw, textvariable=status_var, 
                                  values=["Pending", "Preparing", "Ready", "Delivered", "Cancelled"], 
                                  state="readonly")
        status_menu.pack(ipadx=20, pady=5)
        
        def save_order():
            try:
                room = room_entry.get().strip()
                items = items_entry.get().strip()
                price = price_entry.get().strip()
                status = status_var.get()
                
                if not all([room, items, price, status]):
                    show_error_message("Please fill all fields")
                    return
                
                order_date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                cursor.execute("""INSERT INTO restaurant_orders 
                               (room_number, items, status, total_price, order_date) 
                               VALUES (?, ?, ?, ?, ?)""", 
                             (room, items, status, float(price), order_date))
                conn.commit()
                show_success_message("Order created successfully")
                load_orders()
                # Clear form
                room_entry.delete(0, tk.END)
                items_entry.delete(0, tk.END)
                status_var.set('')
                price_entry.delete(0, tk.END)
            except ValueError:
                show_error_message("Price must be a valid number")
            except Exception as e:
                show_error_message(f"Error: {str(e)}")
        
        tk.Button(nw, text="Save Order", command=save_order, bg="#4CAF50", fg="white", 
                 font=("segoe ui", 11)).pack(pady=20)
    
    def update_status():
        sel = tree.selection()
        if not sel:
            show_error_message("Please select an order")
            return
        
        order_id = tree.item(sel[0])['values'][0]
        
        uw = tk.Toplevel(rw)
        register_window(uw)
        uw.title("Update Order Status")
        uw.geometry("300x200")
        uw.configure(bg="#E6E1E1")
        
        tk.Label(uw, text=f"Order #{order_id}", font=("segoe ui", 12, "bold"), bg="#E6E1E1").pack(pady=10)
        tk.Label(uw, text="New Status:", bg="#E6E1E1").pack(pady=5)
        
        status_var = tk.StringVar()
        status_menu = ttk.Combobox(uw, textvariable=status_var, 
                                  values=["Pending", "Preparing", "Ready", "Delivered", "Cancelled"], 
                                  state="readonly")
        status_menu.pack(ipadx=20, pady=5)
        
        def update():
            try:
                new_status = status_var.get()
                if not new_status:
                    show_error_message("Please select a status")
                    return
                cursor.execute("UPDATE restaurant_orders SET status=? WHERE order_id=?", 
                             (new_status, order_id))
                conn.commit()
                show_success_message("Status updated")
                load_orders()
                # Refresh status display
                status_var.set(new_status)
            except Exception as e:
                show_error_message(f"Error: {str(e)}")
        
        tk.Button(uw, text="Update", command=update, bg="#2196F3", fg="white").pack(pady=15)
    
    # Action buttons
    btn_frame = tk.Frame(rw, bg="#f4f6f8")
    btn_frame.pack(fill="x", padx=10, pady=10)
    tk.Button(btn_frame, text="New Order", command=new_order, bg="#4CAF50", fg="white").pack(side="left", padx=5)
    tk.Button(btn_frame, text="Update Status", command=update_status, bg="#2196F3", fg="white").pack(side="left", padx=5)
    tk.Button(btn_frame, text="View Details", bg="#FF9800", fg="white").pack(side="left", padx=5)

def reviews_window():
    rv = tk.Toplevel(root)
    register_window(rv)
    rv.title("Customer Satisfaction & Feedback")
    rv.geometry("950x650")
    rv.configure(bg="#f4f6f8")
    
    tk.Label(rv, text="Customer Satisfaction & Feedback", font=("segoe ui", 16, "bold"), bg="#f4f6f8").pack(pady=10)
    
    # Feedback table
    cols = ("feedback_id", "booking_id", "guest_name", "type", "rating", "date")
    tree = ttk.Treeview(rv, columns=cols, show='headings', height=15)
    for c in cols:
        tree.heading(c, text=c.replace('_', ' ').title())
        tree.column(c, width=140)
    tree.pack(fill='both', expand=True, padx=10, pady=10)
    
    def load_feedback():
        for r in tree.get_children():
            tree.delete(r)
        try:
            cursor.execute("""
                SELECT feedback_id, booking_id, guest_name, feedback_type, rating, created_at
                FROM customer_feedback
                ORDER BY created_at DESC
            """)
            for row in cursor.fetchall():
                tree.insert('', 'end', values=row)
        except Exception as e:
            show_error_message(f"Error: {str(e)}")
    
    def view_feedback():
        sel = tree.selection()
        if not sel:
            show_error_message("Please select a feedback")
            return
        item = tree.item(sel[0])['values']
        feedback_id = item[0]
        
        try:
            cursor.execute("SELECT * FROM customer_feedback WHERE feedback_id=?", (feedback_id,))
            fb = cursor.fetchone()
            if not fb:
                return
            
            vw = tk.Toplevel(rv)
            register_window(vw)
            vw.title(f"Feedback #{feedback_id}")
            vw.geometry("500x450")
            
            info_text = f"""Feedback ID: {fb[0]}
Booking ID: {fb[1]}
Guest Name: {fb[2]}
Type: {fb[3]}
Rating: {fb[4]} / 5
Message: {fb[5]}
Date: {fb[6]}"""
            
            tk.Label(vw, text=info_text, font=("segoe ui", 10), justify="left").pack(anchor="w", padx=15, pady=15)
        except Exception as e:
            show_error_message(f"Error: {str(e)}")
    
    def add_feedback():
        aw = tk.Toplevel(rv)
        register_window(aw)
        aw.title("Add Customer Feedback")
        aw.geometry("500x550")
        aw.configure(bg="#f4f6f8")
        
        tk.Label(aw, text="Add Customer Feedback", font=("segoe ui", 12, "bold"), bg="#f4f6f8").pack(pady=10)
        
        tk.Label(aw, text="Booking ID", bg="#f4f6f8").pack(anchor="w", padx=15, pady=(10, 0))
        booking_entry = tk.Entry(aw)
        booking_entry.pack(padx=15, fill="x", pady=5)
        
        tk.Label(aw, text="Guest Name", bg="#f4f6f8").pack(anchor="w", padx=15, pady=(10, 0))
        name_entry = tk.Entry(aw)
        name_entry.pack(padx=15, fill="x", pady=5)
        
        tk.Label(aw, text="Feedback Type", bg="#f4f6f8").pack(anchor="w", padx=15, pady=(10, 0))
        type_var = ttk.Combobox(aw, values=["Service Quality", "Cleanliness", "Food Quality", "Staff Behavior", "Room Comfort", "Value for Money", "General"], state="readonly")
        type_var.pack(padx=15, fill="x", pady=5)
        
        tk.Label(aw, text="Rating (1-5)", bg="#f4f6f8").pack(anchor="w", padx=15, pady=(10, 0))
        rating_var = ttk.Combobox(aw, values=["1", "2", "3", "4", "5"], state="readonly")
        rating_var.set("5")
        rating_var.pack(padx=15, fill="x", pady=5)
        
        tk.Label(aw, text="Message", bg="#f4f6f8").pack(anchor="w", padx=15, pady=(10, 0))
        msg_text = tk.Text(aw, height=8, width=50)
        msg_text.pack(padx=15, pady=5, fill="both", expand=True)
        
        def save():
            try:
                booking_id = booking_entry.get().strip()
                guest_name = name_entry.get().strip()
                fb_type = type_var.get()
                rating = int(rating_var.get()) if rating_var.get() else None
                message = msg_text.get("1.0", tk.END).strip()
                
                if not all([booking_id, guest_name, fb_type, message]):
                    show_error_message("Please fill all fields")
                    return
                
                ts = datetime.datetime.utcnow().isoformat()
                cursor.execute("""
                    INSERT INTO customer_feedback
                    (booking_id, guest_name, feedback_type, message, rating, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (booking_id, guest_name, fb_type, message, rating, ts))
                conn.commit()
                create_notification(current_user, f"💬 New feedback from {guest_name}: {fb_type}")
                aw.destroy()
                load_feedback()
            except Exception as e:
                show_error_message(f"Error: {str(e)}")
        
        tk.Button(aw, text="Save Feedback", command=save, bg="#4CAF50", fg="white").pack(pady=15)
    
    btnf = tk.Frame(rv, bg="#f4f6f8")
    btnf.pack(fill='x', padx=10, pady=10)
    tk.Button(btnf, text="View Feedback", command=view_feedback, bg="#2196F3", fg="white").pack(side='left', padx=5)
    tk.Button(btnf, text="Add Feedback", command=add_feedback, bg="#4CAF50", fg="white").pack(side='left', padx=5)
    
    load_feedback()

def loyalty_window():
    lv = tk.Toplevel(root)
    register_window(lv)
    lv.title("Loyalty Program Management")
    lv.geometry("1000x650")
    lv.configure(bg="#f4f6f8")
    
    tk.Label(lv, text="Loyalty Program Management", font=("segoe ui", 16, "bold"), bg="#f4f6f8").pack(pady=10)
    
    # Create notebook for tabs
    notebook = ttk.Notebook(lv)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Members tab
    members_frame = tk.Frame(notebook, bg="#f4f6f8")
    notebook.add(members_frame, text="Members")
    
    # Members table
    cols = ("member_id", "name", "tier", "points", "amount_spent", "joined")
    tree = ttk.Treeview(members_frame, columns=cols, show='headings', height=12)
    for c in cols:
        tree.heading(c, text=c.replace('_', ' ').title())
        tree.column(c, width=140)
    tree.pack(fill='both', expand=True, padx=10, pady=10)
    
    def load_members():
        for r in tree.get_children():
            tree.delete(r)
        try:
            cursor.execute("SELECT member_id, guest_name, tier, points, amount_spent, joined_date FROM loyalty_members ORDER BY points DESC")
            for row in cursor.fetchall():
                tree.insert('', 'end', values=row)
        except Exception as e:
            show_error_message(f"Error: {str(e)}")
    
    def add_member():
        aw = tk.Toplevel(lv)
        register_window(aw)
        aw.title("Add Loyalty Member")
        aw.geometry("400x300")
        
        tk.Label(aw, text="Guest Name").pack(pady=5)
        name_entry = tk.Entry(aw)
        name_entry.pack(padx=10, fill="x")
        
        tk.Label(aw, text="Tier").pack(pady=5)
        tier_var = ttk.Combobox(aw, values=["Bronze", "Silver", "Gold", "Platinum"], state="readonly")
        tier_var.pack(padx=10, fill="x")
        
        tk.Label(aw, text="Starting Points").pack(pady=5)
        points_entry = tk.Entry(aw)
        points_entry.insert(0, "0")
        points_entry.pack(padx=10, fill="x")
        
        def save():
            try:
                name = name_entry.get().strip()
                tier = tier_var.get()
                points = int(points_entry.get())
                
                if not name or not tier:
                    show_error_message("Please fill all fields")
                    return
                
                ts = get_iso_timestamp()
                cursor.execute("""
                    INSERT INTO loyalty_members (guest_name, tier, points, amount_spent, joined_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, tier, points, 0, ts))
                conn.commit()
                create_notification(current_user, f"🎁 Loyalty member {name} added to {tier} tier")
                aw.destroy()
                load_members()
            except Exception as e:
                show_error_message(f"Error: {str(e)}")
        
        tk.Button(aw, text="Save", command=save, bg="#4CAF50", fg="white").pack(pady=15)
    
    def award_points():
        sel = tree.selection()
        if not sel:
            show_error_message("Please select a member")
            return
        member_id = tree.item(sel[0])['values'][0]
        
        aw = tk.Toplevel(lv)
        register_window(aw)
        aw.title("Award Points")
        aw.geometry("400x250")
        
        tk.Label(aw, text="Points to Award").pack(pady=10)
        points_entry = tk.Entry(aw)
        points_entry.pack(padx=10, fill="x", pady=5)
        
        tk.Label(aw, text="Reason").pack(pady=10)
        reason_entry = tk.Entry(aw)
        reason_entry.pack(padx=10, fill="x", pady=5)
        
        def save():
            try:
                points = int(points_entry.get())
                reason = reason_entry.get().strip()
                
                cursor.execute("SELECT guest_name, points FROM loyalty_members WHERE member_id=?", (member_id,))
                member = cursor.fetchone()
                if not member:
                    show_error_message("Member not found")
                    return
                
                new_points = member[1] + points
                cursor.execute("UPDATE loyalty_members SET points=? WHERE member_id=?", (new_points, member_id))
                conn.commit()
                create_notification(current_user, f"🎁 {points} points awarded to {member[0]} - {reason}")
                aw.destroy()
                load_members()
            except Exception as e:
                show_error_message(f"Error: {str(e)}")
        
        tk.Button(aw, text="Award", command=save, bg="#4CAF50", fg="white").pack(pady=10)
    
    members_btn_frame = tk.Frame(members_frame, bg="#f4f6f8")
    members_btn_frame.pack(fill="x", padx=10, pady=5)
    tk.Button(members_btn_frame, text="Add Member", command=add_member, bg="#4CAF50", fg="white").pack(side="left", padx=5)
    tk.Button(members_btn_frame, text="Award Points", command=award_points, bg="#FF9800", fg="white").pack(side="left", padx=5)
    
    load_members()
    
    # Offers tab
    offers_frame = tk.Frame(notebook, bg="#f4f6f8")
    notebook.add(offers_frame, text="Special Offers")
    
    tk.Label(offers_frame, text="Special Offers for Loyalty Members (Admin Only)", font=("segoe ui", 12, "bold"), bg="#f4f6f8").pack(pady=10)
    
    offers_cols = ("offer_id", "offer_name", "discount", "tier", "active", "created_by")
    offers_tree = ttk.Treeview(offers_frame, columns=offers_cols, show='headings', height=12)
    for c in offers_cols:
        offers_tree.heading(c, text=c.replace('_', ' ').title())
        offers_tree.column(c, width=140)
    offers_tree.pack(fill='both', expand=True, padx=10, pady=10)
    
    def load_offers():
        for r in offers_tree.get_children():
            offers_tree.delete(r)
        try:
            cursor.execute("SELECT offer_id, offer_name, discount_percentage, valid_for_tier, active, created_by FROM loyalty_offers ORDER BY created_at DESC")
            for row in cursor.fetchall():
                offers_tree.insert('', 'end', values=row)
        except Exception:
            pass
    
    def add_offer():
        if current_user_role != "Admin" and current_user_role != "admin":
            show_error_message("Only admin can create offers")
            return
        
        ow = tk.Toplevel(lv)
        register_window(ow)
        ow.title("Create New Offer")
        ow.geometry("500x400")
        
        tk.Label(ow, text="Offer Name").pack(pady=5)
        name_entry = tk.Entry(ow)
        name_entry.pack(padx=10, fill="x")
        
        tk.Label(ow, text="Description").pack(pady=5)
        desc_entry = tk.Entry(ow)
        desc_entry.pack(padx=10, fill="x")
        
        tk.Label(ow, text="Discount (%)").pack(pady=5)
        discount_entry = tk.Entry(ow)
        discount_entry.pack(padx=10, fill="x")
        
        tk.Label(ow, text="Valid for Tier").pack(pady=5)
        tier_var = ttk.Combobox(ow, values=["Bronze", "Silver", "Gold", "Platinum", "All"], state="readonly")
        tier_var.pack(padx=10, fill="x")
        
        def save():
            try:
                name = name_entry.get().strip()
                desc = desc_entry.get().strip()
                discount = float(discount_entry.get())
                tier = tier_var.get()
                
                if not all([name, desc, discount, tier]):
                    show_error_message("Please fill all fields")
                    return
                
                ts = get_iso_timestamp()
                cursor.execute("""
                    INSERT INTO loyalty_offers 
                    (offer_name, description, discount_percentage, valid_for_tier, active, created_at, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (name, desc, discount, tier, 1, ts, current_user))
                conn.commit()
                create_notification(current_user, f"🎁 New loyalty offer created: {name} - {discount}% off")
                ow.destroy()
                load_offers()
            except Exception as e:
                show_error_message(f"Error: {str(e)}")
        
        tk.Button(ow, text="Create Offer", command=save, bg="#4CAF50", fg="white").pack(pady=15)
    
    def toggle_offer():
        sel = offers_tree.selection()
        if not sel:
            show_error_message("Please select an offer")
            return
        offer_id = offers_tree.item(sel[0])['values'][0]
        try:
            cursor.execute("SELECT active FROM loyalty_offers WHERE offer_id=?", (offer_id,))
            result = cursor.fetchone()
            if result:
                new_active = 0 if result[0] else 1
                cursor.execute("UPDATE loyalty_offers SET active=? WHERE offer_id=?", (new_active, offer_id))
                conn.commit()
                create_notification(current_user, f"🎁 Offer status updated")
                load_offers()
        except Exception as e:
            show_error_message(f"Error: {str(e)}")
    
    offers_btn_frame = tk.Frame(offers_frame, bg="#f4f6f8")
    offers_btn_frame.pack(fill="x", padx=10, pady=5)
    tk.Button(offers_btn_frame, text="Add Offer", command=add_offer, bg="#4CAF50", fg="white").pack(side="left", padx=5)
    tk.Button(offers_btn_frame, text="Toggle Active", command=toggle_offer, bg="#2196F3", fg="white").pack(side="left", padx=5)
    
    load_offers()

def help_center_window():
    hw = tk.Toplevel(root)
    register_window(hw)
    hw.title("Booking & Service Information")
    hw.geometry("750x650")
    hw.configure(bg="#f4f6f8")
    
    tk.Label(hw, text="Booking & Service Information", font=("segoe ui", 16, "bold"), bg="#f4f6f8").pack(pady=10)
    
    # Search bar for booking ID only
    search_frame = tk.Frame(hw, bg="#f4f6f8")
    search_frame.pack(fill="x", padx=20, pady=10)
    tk.Label(search_frame, text="Search Booking ID:", bg="#f4f6f8", font=("segoe ui", 10)).pack(side="left", padx=(0, 10))
    search_entry = tk.Entry(search_frame, bg="white", font=("segoe ui", 10))
    search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
    
    # Results frame
    canvas = tk.Canvas(hw, bg="#f4f6f8", highlightthickness=0)
    scrollbar = tk.Scrollbar(hw, orient="vertical", command=canvas.yview)
    result_frame = tk.Frame(canvas, bg="#f4f6f8")
    
    result_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=result_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    scrollbar.pack(side="right", fill="y")
    
    def search_booking():
        # Clear previous results
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        booking_id = search_entry.get().strip()
        
        if not booking_id:
            tk.Label(result_frame, text="Enter a booking ID to search", bg="#f4f6f8", 
                    fg="#999999", font=("segoe ui", 10)).pack(pady=20)
            return
        
        try:
            cursor.execute("""SELECT booking_id, guest_name, room_number, check_in_date, 
                           check_out_date, status, total_amount FROM guest_bookings 
                           WHERE booking_id = ?""", (booking_id,))
            booking = cursor.fetchone()
            
            if not booking:
                tk.Label(result_frame, text=f"No booking found with ID: {booking_id}", 
                        bg="#f4f6f8", fg="#d32f2f", font=("segoe ui", 10)).pack(pady=20)
                return
            
            # Display booking information
            booking_card = tk.Frame(result_frame, bg="white", relief="solid", bd=1)
            booking_card.pack(fill="x", pady=10, padx=5)
            
            tk.Label(booking_card, text=f"Booking #{booking[0]}", font=("segoe ui", 12, "bold"), 
                    bg="white", fg="#1565C0").pack(anchor="w", padx=10, pady=(10, 5))
            
            info_text = f"""Guest Name: {booking[1]}
Room Number: {booking[2]}
Check-in: {booking[3]}
Check-out: {booking[4]}
Status: {booking[5]}
Total Amount: ${booking[6]}"""
            
            tk.Label(booking_card, text=info_text, font=("segoe ui", 10), bg="white", 
                    justify="left").pack(anchor="w", padx=10, pady=10)
            
            # Service information section
            service_card = tk.Frame(result_frame, bg="white", relief="solid", bd=1)
            service_card.pack(fill="x", pady=10, padx=5)
            
            tk.Label(service_card, text="Hotel Services", font=("segoe ui", 11, "bold"), 
                    bg="white", fg="#1565C0").pack(anchor="w", padx=10, pady=(10, 5))
            
            services_info = """• Check-in/Check-out available at front desk
• Room service available 24/7
• Complimentary breakfast for all guests
• WiFi access throughout the property
• Restaurant & bar available daily
• Housekeeping service upon request"""
            
            tk.Label(service_card, text=services_info, font=("segoe ui", 9), bg="white", 
                    justify="left").pack(anchor="w", padx=10, pady=10)
        
        except Exception as e:
            tk.Label(result_frame, text=f"Error: {str(e)}", bg="#f4f6f8", 
                    fg="#d32f2f", font=("segoe ui", 10)).pack(pady=20)
    
    def on_search_enter(event):
        search_booking()
    
    search_entry.bind("<Return>", on_search_enter)
    
    # Search button
    search_btn = tk.Button(search_frame, text="Search", command=search_booking, 
                          bg="#2196F3", fg="white", font=("segoe ui", 10), relief="flat", cursor="hand2")
    search_btn.pack(side="left", padx=(0, 10))

def validate_password_strength(password: str) -> tuple:
    """Validate password strength. Returns (is_valid, message)"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c in "!@#$%^&*()-_=+[]{}|;:',.<>?/~`" for c in password):
        return False, "Password must contain at least one special character (!@#$%^&..)"
    return True, "Password is strong"


def change_password_window():
    """Allow user to change their password with strength validation"""
    if not current_user:
        show_error_message("No user logged in")
        return
    
    cpw = tk.Toplevel(root)
    register_window(cpw)
    cpw.title("Change Password")
    cpw.geometry("400x350")
    cpw.configure(bg="#E6E1E1")
    cpw.resizable(False, False)
    
    tk.Label(cpw, text="Change Your Password", font=("segoe", 16, "bold"), bg="#E6E1E1", fg="black").pack(pady=10)
    
    tk.Label(cpw, text="Current Password:", font=("segoe", 10), bg="#E6E1E1").pack(ipady=5, pady=5)
    curr_entry = tk.Entry(cpw, show="*", bg="#d9d9d9", font=("segoe", 10))
    curr_entry.pack(ipadx=20, pady=5)
    
    tk.Label(cpw, text="New Password:", font=("segoe", 10), bg="#E6E1E1").pack(ipady=5, pady=5)
    new_entry = tk.Entry(cpw, show="*", bg="#d9d9d9", font=("segoe", 10))
    new_entry.pack(ipadx=20, pady=5)
    
    tk.Label(cpw, text="Confirm New Password:", font=("segoe", 10), bg="#E6E1E1").pack(ipady=5, pady=5)
    conf_entry = tk.Entry(cpw, show="*", bg="#d9d9d9", font=("segoe", 10))
    conf_entry.pack(ipadx=20, pady=5)
    
    strength_label = tk.Label(cpw, text="", font=("segoe", 9), bg="#E6E1E1")
    strength_label.pack(pady=5)
    
    def check_strength(*args):
        pwd = new_entry.get()
        if pwd:
            is_valid, msg = validate_password_strength(pwd)
            if is_valid:
                strength_label.config(text="✓ " + msg, fg="#2E7D32")
            else:
                strength_label.config(text="✗ " + msg, fg="#C62828")
        else:
            strength_label.config(text="")
    
    new_entry.bind("<KeyRelease>", check_strength)
    
    def change_pw():
        curr = curr_entry.get()
        new = new_entry.get()
        conf = conf_entry.get()
        
        if not curr or not new or not conf:
            show_error_message("All fields required")
            return
        
        # verify current password
        try:
            cursor.execute("SELECT password FROM users WHERE username=?", (current_user,))
            row = cursor.fetchone()
            if not row or not verify_password(curr, row[0]):
                show_error_message("Current password is incorrect")
                return
        except Exception as e:
            show_error_message(f"Error verifying password: {e}")
            return
        
        # validate new password strength
        is_valid, msg = validate_password_strength(new)
        if not is_valid:
            show_error_message(msg)
            return
        
        # check match
        if new != conf:
            show_error_message("New passwords do not match")
            return
        
        # hash and update
        try:
            new_hash = hash_password(new)
            cursor.execute("UPDATE users SET password=? WHERE username=?", (new_hash, current_user))
            conn.commit()
            show_success_message("Password changed successfully!")
            cpw.destroy()
        except Exception as e:
            show_error_message(f"Error updating password: {e}")
    
    # bind Enter key for field navigation
    curr_entry.bind("<Return>", lambda e: new_entry.focus())
    new_entry.bind("<Return>", lambda e: conf_entry.focus())
    conf_entry.bind("<Return>", lambda e: change_pw())
    
    tk.Button(cpw, text="Change Password", command=change_pw, bg="#2196F3", fg="white", font=("segoe", 11, "bold")).pack(pady=15)


def switch_user():
    # simple switch: logout then show login
    logout()

def contact_support_window():
    """Open WhatsApp chat with support using phone number 07081476655"""
    support_number = "07081476655"  # Nigerian format
    
    if not confirm_action("Open WhatsApp chat with support?"):
        return
    
    try:
        # Convert Nigerian number to international format
        int_number = "+234" + support_number[1:]  # 07081476655 → +2347081476655
        
        # Try system WhatsApp first (Windows)
        try:
            import subprocess
            import sys
            import platform
            
            if platform.system() == 'Windows':
                # Windows - try WhatsApp desktop app
                subprocess.Popen(['start', f'whatsapp://send?phone={int_number}'], shell=True)
                show_info_message("Opening WhatsApp...")
                return
        except Exception:
            pass
        
        # Fallback to web WhatsApp
        url = f"https://web.whatsapp.com/send?phone={int_number}&text=Hello%20Support"
        webbrowser.open(url)
        show_info_message("Opening WhatsApp in your browser...")
        
    except Exception as e:
        show_error_message(f"Unable to open WhatsApp: {str(e)}")

def settings_window():
    sw = tk.Toplevel(root)
    register_window(sw)
    sw.title("⚙️ Settings & Preferences")
    sw.geometry("500x750")
    sw.configure(bg="#f4f6f8")
    
    # load settings
    try:
        cursor.execute("SELECT value FROM settings WHERE key='auto_refresh_kpis'")
        ar = cursor.fetchone()
        auto_refresh = tk.IntVar(value=1 if ar and ar[0]=='1' else 0)
        cursor.execute("SELECT value FROM settings WHERE key='notifications_enabled'")
        ne = cursor.fetchone()
        notif_enabled = tk.IntVar(value=1 if ne and ne[0]=='1' else 0)
    except Exception:
        auto_refresh = tk.IntVar(value=1)
        notif_enabled = tk.IntVar(value=1)

    def save_settings():
        try:
            cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ("auto_refresh_kpis", '1' if auto_refresh.get() else '0'))
            cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ("notifications_enabled", '1' if notif_enabled.get() else '0'))
            conn.commit()
            show_success_message("Settings saved")
            # apply auto refresh
            if auto_refresh.get():
                start_kpi_refresh()
            else:
                stop_kpi_refresh()
        except Exception as e:
            show_error_message(f"Error saving settings: {e}")
    
    def delete_account():
        """Delete current user account with confirmation"""
        if current_user_role == "admin":
            show_error_message("Cannot delete admin account. Transfer admin role first.")
            return
        
        warn_msg = f"⚠️ DELETE ACCOUNT WARNING!\n\nYou are about to permanently delete account: {current_user}\n\nThis will:\n- Delete your account\n- Clear your messages\n- Remove your profile\n\nThis CANNOT be undone!\n\nContinue?"
        if not confirm_action(warn_msg):
            return
        
        final_confirm = confirm_action("⚠️ FINAL WARNING!\n\nYour account will be permanently deleted!\n\nConfirm?")
        if not final_confirm:
            return
        
        try:
            # Delete user account
            cursor.execute("DELETE FROM users WHERE username=?", (current_user,))
            # Delete all messages involving this user
            cursor.execute("DELETE FROM messages WHERE sender=? OR receiver=?", (current_user, current_user))
            # Delete notifications
            cursor.execute("DELETE FROM notifications WHERE username=?", (current_user,))
            # Delete profile image if exists
            profile_path = f"profiles/{current_user}.png"
            if os.path.exists(profile_path):
                os.remove(profile_path)
            conn.commit()
            
            show_success_message("Account deleted successfully.")
            sw.destroy()
            # Logout and return to login
            logout()
        except Exception as e:
            show_error_message(f"Error deleting account: {str(e)}")
    
    def factory_reset():
        if current_user_role != "admin":
            show_error_message("Only admins can perform factory reset")
            return
        
        warn_msg = "⚠️ FACTORY RESET WARNING!\n\nThis will DELETE ALL:\n- Guest Bookings\n- Invoices\n- Messages\n- Housekeeping Tasks\n- Maintenance Issues\n- Customer Feedback\n- Loyalty Members\n- Restaurant Orders\n- All Notifications\n\nThis CANNOT be undone!\n\nContinue?"
        if not confirm_action(warn_msg):
            return
        
        final_confirm = confirm_action("⚠️ This is your FINAL WARNING!\n\nAll data will be permanently deleted!\n\nType to confirm you understand.")
        if not final_confirm:
            return
        
        try:
            # Delete all KPI activities
            cursor.execute("DELETE FROM guest_bookings")
            cursor.execute("DELETE FROM invoices")
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM housekeeping_tasks")
            cursor.execute("DELETE FROM maintenance_issues")
            cursor.execute("DELETE FROM customer_feedback")
            cursor.execute("DELETE FROM loyalty_members")
            cursor.execute("DELETE FROM restaurant_orders")
            cursor.execute("DELETE FROM notifications")
            conn.commit()
            
            # Notify all users
            cursor.execute("SELECT username FROM users WHERE role != 'admin'")
            for user in cursor.fetchall():
                create_notification(user[0], "🔄 System has been factory reset by admin. All data cleared.")
            
            show_success_message("Factory Reset Completed!\nAll KPI activities and notifications have been cleared.")
            sw.destroy()
        except Exception as e:
            show_error_message(f"Error during factory reset: {e}")

    # Application Settings Section
    tk.Label(sw, text="Application Settings", font=("segoe ui", 12, "bold"), bg="#f4f6f8", fg="#1a3a52").pack(anchor='w', pady=(10, 5), padx=10)
    tk.Checkbutton(sw, text="Auto-refresh KPI cards", variable=auto_refresh, bg="#f4f6f8", font=("segoe ui", 10)).pack(anchor='w', pady=5, padx=10)
    tk.Checkbutton(sw, text="Enable notifications", variable=notif_enabled, bg="#f4f6f8", font=("segoe ui", 10)).pack(anchor='w', pady=5, padx=10)
    tk.Button(sw, text="💾 Save Settings", command=save_settings, bg="#2196F3", fg="white", font=("segoe ui", 10, "bold"), relief="raised", bd=2).pack(pady=15, padx=10, fill='x')
    
    # Account Management Section
    tk.Label(sw, text="Account Management", font=("segoe ui", 12, "bold"), bg="#f4f6f8", fg="#1a3a52").pack(anchor='w', pady=(20, 5), padx=10)
    tk.Button(sw, text="🔐 Change Password", command=lambda: show_info_message("Password change feature coming soon"), bg="#FF9800", fg="white", font=("segoe ui", 10, "bold"), relief="raised", bd=2).pack(pady=8, padx=10, fill='x')
    tk.Button(sw, text="❌ Delete Account", command=delete_account, bg="#f44336", fg="white", font=("segoe ui", 10, "bold"), relief="raised", bd=2).pack(pady=8, padx=10, fill='x')
    
    # Admin Tools Section
    if current_user_role == "admin":
        tk.Label(sw, text="Admin Tools", font=("segoe ui", 12, "bold"), bg="#f4f6f8", fg="#d32f2f").pack(anchor='w', pady=(20, 5), padx=10)
        tk.Button(sw, text="🔄 Factory Reset", command=factory_reset, bg="#d32f2f", fg="white", font=("segoe ui", 10, "bold"), relief="raised", bd=2).pack(pady=8, padx=10, fill='x')
    
    root.mainloop()


# main content area - initially hidden until login
main_content = tk.Frame(root, bg="#f4f6f8")
# Don't pack yet - will be packed after successful login

tk.Label(main_content,
         text="Dashboard Overview",
         font=("segoe ui",18),
         bg="#f4f6f8",
         fg="#1a1a1a"
         ).pack(anchor="nw",padx=30,pady=20)

# scrollable frame for cards
canvas = tk.Canvas(main_content, bg="#f4f6f8", highlightthickness=0)
scrollbar = tk.Scrollbar(main_content, orient="vertical", bg="#d1dd5d", activerelief="flat", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#f4f6f8")
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# mouse wheel scrolling
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
canvas.bind( lambda e:
    canvas.bind("<MouseWheel>", _on_mousewheel))
canvas.bind(lambda e: 
    canvas.unbind("<MouseWheel>"))

def dashboard_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", dashboard_mousewheel))
canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))


# cards size 250x130
CARD_WIDTH = 255
CARD_HEIGHT = 120
CARD_GAP = 15
COLUMNS_PER_ROW = 4

# texts colors constants
TEXT_COLOR_PRIMARY = "#000000"
TEXT_COLOR_SECONDARY = "#555555"
TEXT_MUTED = "#777777"

# cards background colors
CARD_DEFAULT = "#FFFFFF"
CARD_GREEN = "#E8F5E9"
CARD_BLUE = "#E3F2FD"
CARD_ORANGE = "#FFF3E0"
CARD_TEAL = "#E1F5FE"
CARD_GRAY = "#F5F5F5"

# border colors
BORDER_LIGHT = "#E0E0E0"
BORDER_DARK = "#BDBDBD"

# status colors
SUCCESS_COLOR = "#2E7D32"
WARNING_COLOR = "#EF6C00"
ERROR_COLOR = "#C62828"
INFO_COLOR = "#1565C0"



# cards frame
cards_frame = tk.Frame(scrollable_frame, bg="#f4f6f8")
cards_frame.pack(anchor="nw", fill="x", padx=30,pady=20)

for col in range(COLUMNS_PER_ROW):
    cards_frame.grid_columnconfigure(col, minsize=CARD_WIDTH, weight=0)
    
    for row in range(5):
        cards_frame.grid_rowconfigure(row, minsize=CARD_HEIGHT, weight=0)
        
def cards_info(parent, title, value, row, col, bg="#ffffff", fg="#1a1a1a", icon=None):
    card = tk.Frame(
        parent,
        bg=bg,
        width=CARD_WIDTH,
        height=CARD_HEIGHT,
        relief="flat",
        cursor="hand2"
    )
    card.grid(row=row, column=col, padx=CARD_GAP, pady=CARD_GAP, sticky="nsew")
    card.grid_propagate(False)

    tk.Label(
        card,
        text=f"{icon} {title}",
        font=("segoe ui", 11),
        bg=bg,
        fg=fg
    ).place(x=15, y=15)

    value_label = tk.Label(
        card,
        text=value,
        font=("segoe ui", 20, "bold"),
        bg=bg,
        fg=fg
    )
    value_label.place(x=15, y=55)
    global kpi_labels
    kpi_labels[title] = value_label

    return card




# ===== DASHBOARD KPI CARDS =====

# Row 1
cards_info(cards_frame, "Today's Revenue", f"${revenue}", 0, 0, "#2ECC71", fg="#FFFFFF", icon="💰")
cards_info(cards_frame, "Total Rooms", str(total_rooms), 0, 1, "#4A90E2", fg="#FFFFFF", icon="🏨")
cards_info(cards_frame, "Occupied rooms", str(occupied_rooms), 0, 2, "#F5A623", fg="#FFFFFF", icon="🛏")
cards_info(cards_frame, "Today Checkins", str(today_checkins), 0, 3, "#50E3C2", fg="#FFFFFF", icon="📅")

# Row 2
cards_info(cards_frame, "Available Rooms", str(available_rooms), 1, 0, "#FFFFFF", fg="#1a1a1a", icon="⏱")
cards_info(cards_frame, "Customer Satisfaction", "90%", 1, 1, "#49A79B", fg="#ffffff", icon="⭐")
loyalty_count = get_count("SELECT COUNT(*) FROM loyalty_members")
cards_info(cards_frame, "Loyalty Members", str(loyalty_count), 1, 2, "#FFFFFF", fg="#1a1a1a", icon="🎁")
active_offers = get_count("SELECT COUNT(*) FROM loyalty_offers WHERE active=1")
cards_info(cards_frame, "Special Offers", f"{active_offers} Active", 1, 3, "#FFFFFF", fg="#1a1a1a", icon="🎯")

# Row 3
cards_info(cards_frame, "Pending payments", str(pending_payments), 2, 0, "#27775F", fg="#ffffff", icon="🟢")
rooms_needing_cleaning = get_count("SELECT COUNT(*) FROM housekeeping_tasks WHERE status='Pending'")
cards_info(cards_frame, "Rooms Needing Cleaning", str(rooms_needing_cleaning), 2, 1, "#FFFFFF", fg="#000000", icon="🧹")
maintenance_open = get_count("SELECT COUNT(*) FROM maintenance_issues WHERE status!='Fixed'")
cards_info(cards_frame, "Maintenance Issues", f"{maintenance_open} Open", 2, 2, "#FAB23F", fg="#ffffff", icon="🔧")
cards_info(cards_frame, "Average stay", average_stay, 2, 3, "#57BA3C", fg="#ffffff", icon="💳")

# wire KPI cards to open related windows on click
try:
    kpi_labels['Total Rooms'].master.bind('<Button-1>', lambda e: rooms_window())
    kpi_labels['Available Rooms'].master.bind('<Button-1>', lambda e: rooms_window())
    kpi_labels['Today Checkins'].master.bind('<Button-1>', lambda e: bookings_window())
except Exception:
    pass

# Build sidebar with section frames and ordered menu
ensure_profiles_dir()

# section frames with consistent spacing
ops_frame = tk.Frame(side_bar, bg=COLOR_SIDEBAR)
ops_frame.pack(fill="x", pady=(10, 5))
tk.Label(ops_frame, text="— Hotel Operations —", fg=COLOR_GOLD, bg=COLOR_SIDEBAR, font=("segoe ui", 9, "bold")).pack(fill="x", pady=(5, 8))

mgmt_frame = tk.Frame(side_bar, bg=COLOR_SIDEBAR)
mgmt_frame.pack(fill="x", pady=(10, 5))
tk.Label(mgmt_frame, text="— Management —", fg=COLOR_GOLD, bg=COLOR_SIDEBAR, font=("segoe ui", 9, "bold")).pack(fill="x", pady=(5, 8))

support_frame = tk.Frame(side_bar, bg=COLOR_SIDEBAR)
support_frame.pack(fill="x", pady=(10, 5))
tk.Label(support_frame, text="— Support —", fg=COLOR_GOLD, bg=COLOR_SIDEBAR, font=("segoe ui", 9, "bold")).pack(fill="x", pady=(5, 8))

account_frame = tk.Frame(side_bar, bg=COLOR_SIDEBAR)
account_frame.pack(fill="x", pady=(10, 5))
tk.Label(account_frame, text="— Account —", fg=COLOR_GOLD, bg=COLOR_SIDEBAR, font=("segoe ui", 9, "bold")).pack(fill="x", pady=(5, 8))

# menu definitions in desired order (section, text, roles, command_name)
menu_defs = [
    (ops_frame, "Dashboard", ["admin", "staff", "receptionist"], "dashboard_window"),
    (ops_frame, "Manage Rooms", ["admin"], "rooms_window"),
    (ops_frame, "Manage Bookings", ["admin", "staff"], "bookings_window"),
    (ops_frame, "Guests", ["admin", "staff"], "guests_window"),
    (ops_frame, "Housekeeping", ["admin", "staff"], "housekeeping_window"),
    (ops_frame, "Maintenance", ["admin", "staff"], "maintenance_window"),
    (ops_frame, "Invoices & Payments", ["admin"], "invoices_window"),
    (ops_frame, "Restaurant / Room Service", ["admin", "staff"], "restaurant_window"),
    (mgmt_frame, "Reports", ["admin"], "reports_window"),
    (mgmt_frame, "Reviews & Feedback", ["admin"], "reviews_window"),
    (mgmt_frame, "Loyalty Program", ["admin"], "loyalty_window"),
    (mgmt_frame, "Settings", ["admin"], "settings_window"),
    (support_frame, "Help Center", ["admin", "staff", "receptionist"], "help_center_window"),
    (support_frame, "Contact Support", ["admin", "staff", "receptionist"], "contact_support_window"),
    (account_frame, "Recent Users", ["admin", "staff", "receptionist"], "recent_users_window"),
    (account_frame, "Switch User", ["admin", "staff", "receptionist"], "switch_user"),
    (account_frame, "Change Password", ["admin", "staff", "receptionist"], "change_password_window"),
    (account_frame, "Logout", ["admin", "staff", "receptionist"], "logout"),
]

# create and store buttons - directly in their parent frames
for parent, text, roles, cmd_name in menu_defs:
    cmd = globals().get(cmd_name)
    if not callable(cmd):
        # fallback to info popup if not implemented yet
        def make_stub(n):
            return lambda: show_info_message(f"{n} not implemented yet")
        cmd = make_stub(text)
    # create button directly in the parent section frame
    btn = tk.Button(parent, text=text, font=("segoe ui", 11), fg="#ffffff", bg=COLOR_SIDEBAR, activebackground=COLOR_SIDEBAR_HOVER, activeforeground="white", relief="flat", anchor="w", padx=20, cursor="hand2", command=cmd)
    # normalize allowed roles to lowercase for consistent matching
    try:
        btn.allowed_roles = [r.lower() for r in roles]
    except Exception:
        btn.allowed_roles = roles
    # hover
    btn.bind("<Enter>", lambda e, b=btn: b.config(bg=COLOR_SIDEBAR_HOVER, fg="white"))
    btn.bind("<Leave>", lambda e, b=btn: b.config(bg=COLOR_SIDEBAR, fg="white"))
    menu_buttons[text] = btn
    # pack immediately in parent frame with consistent spacing
    try:
        btn.pack(fill='x', pady=2)
    except Exception:
        pass

# pack according to role when user logs in via apply_role_permission
apply_role_permission()



# Initialize the application - start with login window

if __name__ == "__main__":
    try:
        root.withdraw()  # Hide main window until login
        face_recognition_window()  # Show face recognition window first
        root.mainloop()  # Start the event loop
    except Exception as e:
        print(f"Fatal error: {e}")