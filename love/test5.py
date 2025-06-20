import cv2 as cv
import pyaudio
import wave
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import parselmouth
import queue
import threading
import matplotlib.pyplot as plt
from colorama import Fore, Style
import speech_recognition as sr
import time
import sqlite3

# === CONSTANTS ===
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OUTPUT_FILENAME = "output.wav"
TRANSCRIPT_FILENAME = "transcript.txt"
VIDEO_FILENAME = "output.avi"
REPORT_FILENAME = "report.txt"

# === INITIALIZATIONS ===
audio_queue = queue.Queue()
face_detect = cv.CascadeClassifier('pretrained.xml')
cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out_video = cv.VideoWriter(VIDEO_FILENAME, fourcc, 20.0, (640, 480))

recording = True
pitch_values_list = []
expression_counts = {"happy": 0, "neutral": 0, "sad": 0, "fear": 0, "nervous": 0, "surprise": 0}

# === FUNCTION DEFINITIONS ===
def analyze_audio():
    try:
        sound = parselmouth.Sound(OUTPUT_FILENAME)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=300)
        pitch_values = pitch.selected_array['frequency']
        voiced_pitch = pitch_values[(pitch_values > 75) & (pitch_values < 300)]
        avg_pitch = np.mean(voiced_pitch) if len(voiced_pitch) > 0 else 0
        pitch_values_list.append(avg_pitch)
        print(Style.BRIGHT + Fore.CYAN + f"Average Voiced Pitch: {avg_pitch:.2f} Hz" + Style.RESET_ALL)
        if avg_pitch < 100:
            print(Fore.RED + "Low pitch detected → Possible low confidence!" + Style.RESET_ALL)
        else:
            print(Fore.GREEN + "Good pitch range → Confident speech!" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Audio analysis error: {e}" + Style.RESET_ALL)

def analyze_expression(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        print(Style.BRIGHT + Fore.YELLOW + f"Detected Expression: {emotion}" + Style.RESET_ALL)
        if emotion in expression_counts:
            expression_counts[emotion] += 1
        else:
            expression_counts["surprise"] += 1
        return emotion
    except Exception as e:
        print(Fore.RED + f"Expression detection error: {e}" + Style.RESET_ALL)
        return "surprise"

# def compute_overall_score():
    # happy_score = expression_counts.get("happy", 0)
    # neutral_score = expression_counts.get("neutral", 0)
    # nervous_score = sum(expression_counts[k] for k in ['sad', 'fear', 'nervous'])
    # avg_pitch = np.mean(pitch_values_list) if pitch_values_list else 0
    # pitch_confidence = 1 if avg_pitch > 120 else 0
    # emotion_confidence = 1 if happy_score + neutral_score > nervous_score else 0
    # combined_confidence = pitch_confidence + emotion_confidence
    # if combined_confidence == 2:
    #     score = np.random.randint(85, 96)
    #     level = "High Confidence"
    # elif combined_confidence == 1:
    #     score = np.random.randint(65, 76)
    #     level = "Moderate Confidence"
    # else:
    #     score = np.random.randint(45, 56)
    #     level = "Low Confidence"
    # print(Style.BRIGHT + Fore.GREEN + f"\n🎯 Confidence Score: {score}/100 → {level}" + Style.RESET_ALL)
    # return score, level
def compute_overall_score():
    happy_score = expression_counts.get("happy", 0)
    neutral_score = expression_counts.get("neutral", 0)
    sad_score = expression_counts.get("sad", 0)
    fear_score = expression_counts.get("fear", 0)
    nervous_score = expression_counts.get("nervous", 0)
    
    total_positive = happy_score + neutral_score
    total_negative = sad_score + fear_score + nervous_score
    
    avg_pitch = np.mean(pitch_values_list) if pitch_values_list else 0
    pitch_confidence = 1 if avg_pitch > 120 else 0

    # Emotion confidence is strictly based on positive vs negative count
    if total_negative > total_positive:
        emotion_confidence = 0
    elif total_positive > total_negative:
        emotion_confidence = 1
    else:
        emotion_confidence = 0  # In case of tie, consider low confidence

    combined_confidence = pitch_confidence + emotion_confidence

    # Scoring logic based on combined confidence
    if combined_confidence == 2:
        score = np.random.randint(80, 91)  # High confidence
        level = "High Confidence"
    elif combined_confidence == 1:
        score = np.random.randint(50, 71)  # Moderate confidence
        level = "Moderate Confidence"
    else:
        score = np.random.randint(30, 45)  # Low confidence
        level = "Low Confidence"

    print(Style.BRIGHT + Fore.GREEN + f"\n🎯 Confidence Score: {score}/100 → {level}" + Style.RESET_ALL)
    return score, level


def save_report(score, level):
    with open(REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write("📋 Presentation Evaluation Report\n")
        f.write("=" * 35 + "\n")
        f.write(f"Confidence Score: {score} / 100\n")
        f.write(f"Confidence Level: {level}\n")
        f.write(f"Average Pitch: {np.mean(pitch_values_list):.2f} Hz\n\n")
        f.write("Emotion Distribution:\n")
        for k, v in expression_counts.items():
            f.write(f" - {k.capitalize()}: {v}\n")

def generate_graph(score, level):
    # score, level = compute_overall_score() #this will generate one more confidence score and display it in the terminal as compute_overall_score() is being called
    avg_pitch = np.mean(pitch_values_list) if pitch_values_list else 0
    save_report(score, level)
    labels = list(expression_counts.keys()) + ["Confidence Score", "Avg Pitch"]
    values = list(expression_counts.values()) + [score, avg_pitch]
    colors = ['green' if k in ['happy', 'neutral'] else 'red' for k in expression_counts.keys()] + ['blue', 'purple']
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    bars = plt.bar(labels, values, color=colors)
    plt.title("Comprehensive Performance Analysis", fontsize=14)
    plt.ylabel("Counts / Score / Hz")
    plt.xticks(rotation=45)
    plt.legend(['Green: Positive', 'Red: Negative', 'Blue: Score', 'Purple: Pitch'], loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=8)
    radar_labels = ['happy', 'neutral', 'sad', 'fear', 'nervous', 'surprise', 'Confidence', 'Pitch']
    radar_values = [expression_counts['happy'], expression_counts['neutral'], expression_counts['sad'], expression_counts['fear'], expression_counts['nervous'], expression_counts['surprise'], score, min(100, (avg_pitch / 250) * 100 if avg_pitch else 0)]
    radar_values += radar_values[:1]
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]
    ax = plt.subplot(1, 2, 2, polar=True)
    ax.plot(angles, radar_values, 'b-', linewidth=2)
    ax.fill(angles, radar_values, 'skyblue', alpha=0.4)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=9)
    ax.set_title("Radar Chart: Emotions & Confidence", fontsize=13, pad=20)
    plt.tight_layout()
    plt.show()

def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.AudioFile(OUTPUT_FILENAME) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
            with open(TRANSCRIPT_FILENAME, "w") as f:
                f.write(transcript)
            print(Style.BRIGHT + Fore.MAGENTA + f"Transcription: {transcript}" + Style.RESET_ALL)
        except sr.UnknownValueError:
            print(Fore.RED + "Could not understand audio." + Style.RESET_ALL)
        except sr.RequestError as e:
            print(Fore.RED + f"Speech recognition error: {e}" + Style.RESET_ALL)
# database connecting 
def get_suggestions_from_db(categories):
    conn = sqlite3.connect('resources.db')
    cursor = conn.cursor()
    result = []
    for category in categories:
        cursor.execute("SELECT description, link FROM suggestions WHERE category=?", (category,))
        rows = cursor.fetchall()
        for row in rows:
            result.append({"area": category.capitalize(), "desc": row[0], "resource": row[1]})
    conn.close()
    return result

def analyze_performance(score, level):
    categories = []
    if level == "Low Confidence":
        categories.append("confidence")
    if expression_counts['sad'] + expression_counts['fear'] + expression_counts['nervous'] > expression_counts['happy']:
        categories.append("expression")
    if pitch_values_list and np.mean(pitch_values_list) < 100:
        categories.append("speech")
    suggestions = get_suggestions_from_db(categories)
    if suggestions:
        print(Fore.BLUE + Style.BRIGHT + "\n📘 Suggestions to Improve:\n" + Style.RESET_ALL)
        for s in suggestions:
            print(f"[{s['area']}] {s['desc']} → {s['resource']}")

# === AUDIO RECORDING THREAD ===
def record_audio():
    global recording
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print(Style.BRIGHT + Fore.YELLOW + "Recording... Press 's' to stop." + Style.RESET_ALL)
    frames = []
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_queue.put(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print(Style.BRIGHT + Fore.BLUE + "Recording stopped." + Style.RESET_ALL)
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    analyze_audio()
    transcribe_audio()

# === MAIN EXECUTION ===
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

print(Fore.CYAN + "📚 Emotion recognition via DeepFace [https://github.com/serengil/deepface]" + Style.RESET_ALL)
print(Fore.CYAN + "🎤 Pitch analysis via Parselmouth (Praat) [https://parselmouth.readthedocs.io/]" + Style.RESET_ALL)

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    out_video.write(frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        emotion = analyze_expression(frame[y:y+h, x:x+w])
        cv.putText(frame, f"Expression: {emotion}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elapsed = int(time.time() - start_time)
    cv.putText(frame, f"Time: {elapsed}s", (500, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.imshow("Presentation Evaluation Feed", frame)
    if cv.waitKey(1) & 0xFF == ord("s"):
        recording = False
        break

cap.release()
out_video.release()
cv.destroyAllWindows()
audio_thread.join()
score, level = compute_overall_score()
save_report(score, level)
generate_graph(score, level)
analyze_performance(score, level)