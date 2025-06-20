import sqlite3

# Create or overwrite the database file
conn = sqlite3.connect("resources.db")
cursor = conn.cursor()

# Create the 'suggestions' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    link TEXT NOT NULL
)
''')

# Insert sample suggestion rows
cursor.executemany('''
INSERT INTO suggestions (category, description, link) VALUES (?, ?, ?)
''', [
    ("confidence", "Practice speaking with a friend or in front of a mirror.", "https://www.toastmasters.org"),
    ("speech", "Work on varying your pitch and tone.", "https://speechmatters.org/tips"),
    ("expression", "Improve facial expressiveness by practicing in front of a camera.", "https://www.skillsyouneed.com/ps/body-language-face.html")
])

conn.commit()
conn.close()

print("✅ resources.db has been created with sample data.")