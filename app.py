import os
import requests, base64
from azure.storage.blob import BlobClient
import cv2
import requests
# from suno_api import generate_audio_by_prompt, get_audio_information
from dotenv import load_dotenv
import time
# from summarization import summarize_prompts
from azure.storage.blob import BlobClient
from langchain_openai import ChatOpenAI
from openai import OpenAI





load_dotenv()
azure_connection_string = os.environ.get("AZURE_CONNECTION_STRING")
if not azure_connection_string:
    raise ValueError("No Azure connection string found.")


def make_dir_test():
    try:
        os.mkdir("images")
    except FileExistsError:
        pass


def upload_images(image_path):
    container_name = "harmoniq-imagestorage"
    # Create a BlobServiceClient object
    for i in range(2):
        '''
        have users enter their name to make 
        their images
        uniquely identifyiable 
        '''
        blob_name = f"captured_image_{i + 1}.jpg"
        file_path = f"images/captured_image_{i + 1}.jpg"
        blob_client = BlobClient.from_connection_string(
            conn_str=azure_connection_string,
            container_name=container_name,
            blob_name=blob_name,
        )

        try:
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                print(f"Uploaded {blob_name} successfully.")
                print(f"Blob URL: {blob_client.url}")
                url = blob_client.url
                analyze_image(url)
        except Exception as e:
            print(f"Failed to upload {blob_name}. Error: {e}")
            url = None 
        return url

def analyze_image(url):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key not found.")
        return
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyse the image and describe it."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{url}",
                        },
                    },
                ],
            }
        ],
        max_tokens=250,
    )
    print(response.choices[0].message.content)
    prompt = response.choices[0].message.content
    return prompt


def detect_faces_live(frame, face_cascade):
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier.")
        return []

    # Convert the frame to grayscale for better face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw boxes around the detected faces
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return faces


def capture_images():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    num_pic = 2
    picture_count = 0
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    start_time = time.time()

    while picture_count < num_pic:
        ret, frame = cap.read()

        if not ret:
            print("Error: Cannot receive frame.")
            break

        # Detect faces in the frame
        faces = detect_faces_live(frame, face_cascade)

        if len(faces) > 0 and (time.time() - start_time) >= 2:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                f"Picture {picture_count + 1}",
                (50, 50),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera feed", frame)

            make_dir_test()
            filename = f"images/captured_image_{picture_count + 1}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image captured and saved as '{filename}'")

            picture_count += 1
            start_time = time.time()  # Reset the timer after capturing an image
        else:
            cv2.imshow("Camera feed", frame)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_images()
    image_path = "images/captured_image_2.jpg"
    upload_images(image_path)
    