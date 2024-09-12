from pathlib import Path

from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

from final import *
from videopro import *
from final_image import *
from tkinter.filedialog import askopenfilename

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("704x488")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 488,
    width = 704,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: fintegration(),
    relief="flat"
)
button_1.place(
    x=466.0,
    y=142.99999999999994,
    width=141.0439453125,
    height=46.0
)



button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: vidpro(),
    relief="flat"
)
button_3.place(
    x=466.0,
    y=220.99999999999994,
    width=141.0439453125,
    height=46.0
)


def open_file_and_process():
    file_path = askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    
    if file_path:
        image_check(file_path)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: open_file_and_process(),
    relief="flat"
)

button_2.place(
    x=466.0,
    y=300.99999999999994,
    width=141.0439453125,
    height=46.0
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    197.0,
    257.00000000000006,
    image=image_image_1
)

canvas.create_text(
    485.0,
    57.99999999999994,
    anchor="nw",
    text="SYSTEM",
    fill="#040404",
    font=("Inter Bold", 25 * -1)
)

canvas.create_text(
    400.0,
    20.999999999999943,
    anchor="nw",
    text="SECURITY MONITORING ",
    fill="#040404",
    font=("Inter Bold", 25 * -1)
)
window.resizable(False, False)
window.mainloop()
