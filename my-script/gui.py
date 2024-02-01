from tkinter import *
from tkinter import ttk
from tkinter.font import Font

from src.cyacc import parse


def run():
    print("-------restart-------")
    code = txt.get("1.0", "end").split("\n")
    for data in code:
        if data != "":
            parse(data)


root = Tk()
root.title("Text 1")
root.minsize(100, 100)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Frame
frame1 = ttk.Frame(root, padding=10)
frame1.rowconfigure(1, weight=1)
frame1.columnconfigure(0, weight=1)
frame1.grid(sticky=(N, W, S, E))

# Button
button1 = ttk.Button(frame1, text="RUN", command=run)
button1.grid(row=0, column=0, columnspan=2, sticky=(N, E))

# Text
f = Font(family="Helvetica", size=16)
v1 = StringVar()
txt = Text(frame1, height=15, width=70)
txt.configure(font=f)
txt.grid(row=1, column=0, sticky=(N, W, S, E))

# Scrollbar
scrollbar = ttk.Scrollbar(frame1, orient=VERTICAL, command=txt.yview)
txt["yscrollcommand"] = scrollbar.set
scrollbar.grid(row=1, column=1, sticky=(N, S))

root.mainloop()
