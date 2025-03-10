import pyttsx3

text = ("A futuristic city at sunset, with flying cars, towering skyscrapers, "
        "and neon lights reflecting on glass buildings. The sky is a blend of orange "
        "and purple hues, while people in advanced suits walk along holographic pathways.")

engine = pyttsx3.init()
engine.save_to_file(text, "image_prompt.mp3")
engine.runAndWait()

print("MP3 file generated: image_prompt.mp3")