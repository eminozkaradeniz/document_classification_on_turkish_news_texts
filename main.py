import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import pickle


model = pickle.load(open('model/model.sav', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pk', 'rb'))


def predict_news_report(text: list) -> str:
    text = vectorizer.transform(text)
    pred = model.predict(text)
    return pred[0]


def button_onclick():
    pred = predict_news_report([text_area.get("1.0", tk.END)])
    messagebox.showinfo(title="News Class", message=pred)
    text_area.delete("1.0", tk.END)
    

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("600x450")
    root.title("News Classification")
    
    label = tk.Label(root, text="Enter the news text", font=("Times New Roman", 14))
    label.place(x=20, y=20)
    
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15,
                                          font=("Times New Roman", 14))
    text_area.place(x=20, y=50)
    
    button = tk.Button(root, text="Submit", command=button_onclick)
    button.place(x=20, y=400)
    root.mainloop()
