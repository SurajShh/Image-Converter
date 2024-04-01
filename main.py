import cv2
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from scipy.signal import convolve2d, butter, filtfilt
from scipy.ndimage import gaussian_filter, median_filter

window = Tk()
# add widgets here

window.title('Image Processing')
window.geometry("1000x600+10+20")
white_frame = tk.Frame(bd=0, highlightthickness=0, background="white")
white_frame.place(x=0, y=0, relwidth=1.0, relheight=.10, anchor="nw")
window.config(bg="skyblue")
logo_image = Image.open("C:\\Users\\suraj\\PycharmProject\\DIP_Project_Final\\logo.png")  # Replace "logo.png" with the path to your logo image
logo_image = logo_image.resize((50, 50))
logo_img = ImageTk.PhotoImage(logo_image)
logo_label = Label(window, image=logo_img, bg="skyblue")
logo_label.place(x=440, y=7)
decorative_frame = Frame(window, bg="white", highlightthickness=2, highlightbackground="black")
decorative_frame.place(x=550, y=100, width=400, height=400)
lbl5 = Label(window, text="Upload your Image here ", fg='black', bg='white', font=("Comic Sans MS", 15))
lbl5.place(x=600, y=200)

lbl = Label(window, text="ImageGenius", fg='black', bg='white', font=("Trebuchet MS", 25))
lbl.place(x=500, y=5)
b1 = Button(window, text='Upload File', width=15, bg='#ffb3fe', font=20, command=lambda: upload_file())
b1.place(x=55, y=100)
lbl2 = Label(window, text="Do you want to Continue ?", fg='black', bg='skyblue', font=("Comic Sans MS", 20))
b2 = Button(window, text='YES', width=10, height=1, bg='#ffb3fe', font=20, command=lambda: openNewWindow())
operation_var1 = StringVar()
operation_var1.set("Image Conversion")
operation_var2 = StringVar()
operation_var2.set("Noise")
operation_var3 = StringVar()
operation_var3.set("Filters")
operation_var4 = StringVar()
operation_var4.set("Image Restoration")


def apply_grayscale(image):
    image_pil = ImageTk.getimage(image)
    grayscale_image = image_pil.convert("L")
    return np.array(grayscale_image)


def apply_binary(image):
    grayscale_image = image.convert("L")
    binary_image = grayscale_image.point(lambda x: 0 if x < 128 else 255, '1')
    return binary_image


def add_salt_and_pepper_noise(image, amount):
    image_copy = np.copy(image)
    height, width, _ = image_copy.shape
    num_salt_pixels = int(amount * height * width)
    coordinates = [(np.random.randint(0, height), np.random.randint(0, width)) for _ in range(num_salt_pixels)]

    for coord in coordinates:
        image_copy[coord[0], coord[1]] = [255, 255, 255] if np.random.random() < 0.5 else [0, 0, 0]

    return image_copy


def add_gaussian_noise(image, mean=0, std=300):
    image_array = np.array(image)
    h, w, c = image_array.shape
    noise = np.random.normal(mean, std, (h, w, c))
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def add_poisson_noise(image):
    image_array = np.array(image)
    noisy_image = np.random.poisson(image_array).astype(np.uint8)
    return Image.fromarray(noisy_image)


def apply_high_pass_filter(image):
    image_array = np.array(image)
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    filtered_image = np.zeros_like(image_array)

    for channel in range(image_array.shape[2]):
        filtered_image[:, :, channel] = convolve2d(image_array[:, :, channel], kernel, mode='same', boundary='symm')

    # Scale the filtered image back to the 0-255 range
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return Image.fromarray(filtered_image)


def apply_median_filter(image, size):
    image_array = np.array(image)
    filtered_image = np.zeros_like(image_array)

    for channel in range(image_array.shape[2]):
        filtered_image[:, :, channel] = median_filter(image_array[:, :, channel], size=size)

    return Image.fromarray(filtered_image)


def apply_low_pass_filter(image, sigma):
    image_array = np.array(image)
    filtered_image = gaussian_filter(image_array, sigma=sigma)

    # Scale the filtered image back to the 0-255 range
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return Image.fromarray(filtered_image)


def apply_butterworth_filter(image, cutoff, order):
    image_array = np.array(image)

    # Normalize the cutoff frequency to the range [0, 1]
    cutoff_norm = cutoff / 0.5 * 0.25

    # Design the Butterworth filter
    b, a = butter(order, cutoff_norm, btype='low')

    filtered_image = np.zeros_like(image_array)

    for channel in range(image_array.shape[2]):
        filtered_image[:, :, channel] = filtfilt(b, a, image_array[:, :, channel])

    return Image.fromarray(np.clip(filtered_image, 0, 255).astype(np.uint8))


def display_image(image, title):
    if isinstance(image, np.ndarray):  # If it's a NumPy array
        img_resized = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((600, 600))
    else:  # If it's a PIL image
        img_resized = image.resize((600, 600))

    img_tk = ImageTk.PhotoImage(img_resized)

    if title == "Grayscale Image":
        image_label = tk.Label(new_window, image=img_tk)
        image_label.image = img_tk
        image_label.place(x=625, y=80)
    elif title == "Binary Image":
        binary_label = tk.Label(new_window, image=img_tk)
        binary_label.image = img_tk
        binary_label.place(x=625, y=80)
    elif title == "Salt & Pepper Noise":
        noise_label = tk.Label(new_window, image=img_tk)
        noise_label.image = img_tk
        noise_label.place(x=625, y=80)
    elif title == "Gaussian Noise":
        noise_label = tk.Label(new_window, image=img_tk)
        noise_label.image = img_tk
        noise_label.place(x=625, y=80)
    elif title == "Poisson Noise":
        noise_label = tk.Label(new_window, image=img_tk)
        noise_label.image = img_tk
        noise_label.place(x=625, y=80)
    elif title == "High Pass Filter":
        filter_label = tk.Label(new_window, image=img_tk)
        filter_label.image = img_tk
        filter_label.place(x=625, y=80)
    elif title == "Median Filter":
        filter_label = tk.Label(new_window, image=img_tk)
        filter_label.image = img_tk
        filter_label.place(x=625, y=80)
    elif title == "Low Pass Filter":
        filter_label = tk.Label(new_window, image=img_tk)
        filter_label.image = img_tk
        filter_label.place(x=625, y=80)
    elif title == "Butterworth Filter":
        filter_label = tk.Label(new_window, image=img_tk)
        filter_label.image = img_tk
        filter_label.place(x=625, y=80)


def apply_operation():
    selected_operation1 = operation_var1.get()
    selected_operation2 = operation_var2.get()
    selected_operation3 = operation_var3.get()
    if selected_operation1 == "Grayscale":
        grayscale_image = apply_grayscale(img1)
        display_image(grayscale_image, "Grayscale Image")
    elif selected_operation1 == "Binary":
        binary_image = apply_binary(original_image)
        display_image(binary_image, "Binary Image")
    elif selected_operation2 == "Salt & Pepper":
        noise_amount = 0.05  # Set the noise amount here (you can adjust this value)
        noisy_image = add_salt_and_pepper_noise(original_image, noise_amount)
        display_image(noisy_image, "Salt & Pepper Noise")
    elif selected_operation2 == "Gaussian":
        noisy_image = add_gaussian_noise(original_image, mean=0, std=25)  # You can adjust the mean and std as needed
        display_image(noisy_image, "Gaussian Noise")
    elif selected_operation2 == "Poisson":
        noisy_image = add_poisson_noise(original_image)
        display_image(noisy_image, "Poisson Noise")
    elif selected_operation3 == "High pass filter":
        highpass_image = apply_high_pass_filter(original_image)
        display_image(highpass_image, "High Pass Filter")
    elif selected_operation3 == "Median filter":
        filter_size = 3  # You can adjust the filter size (e.g., 3, 5, 7, etc.)
        median_image = apply_median_filter(original_image, size=filter_size)
        display_image(median_image, "Median Filter")
    elif selected_operation3 == "Low pass filter":
        sigma = 3  # You can adjust the sigma value to control the amount of blurring
        lops_image = apply_low_pass_filter(original_image, sigma)
        display_image(lops_image, "Low Pass Filter")
    elif selected_operation3 == "Butterworth filter":
        cutoff_frequency = 0.1  # You can adjust the cutoff frequency (0.0 to 1.0)
        filter_order = 8  # You can adjust the filter order (e.g., 1, 2, 3, 4, etc.)
        butterworth_image = apply_butterworth_filter(original_image, cutoff_frequency, filter_order)
        display_image(butterworth_image, "Butterworth Filter")


def upload_file():
    global img, img1, original_image
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    img1 = Image.open(filename)
    original_image = img1.copy()  # Store the original image for other operations
    img_resized = img.resize((400, 400))  # new width & height
    img_resized1 = img1.resize((300, 300))
    img = ImageTk.PhotoImage(img_resized)
    img1 = ImageTk.PhotoImage(img_resized1)

    image_label = tk.Label(window, image=img)
    image_label.place(x=550, y=100)  # Set the desired position of the image label
    lbl2.place(x=80, y=550)
    b2.place(x=500, y=550)


def openNewWindow():
    global new_window
    new_window = Toplevel(window)
    new_window.title("Image Processing")
    new_window.geometry("600x600")
    left_frame = tk.Frame(new_window, bg="skyblue", width=700)
    left_frame.pack(side="right", fill="y")
    image_label1 = tk.Label(new_window, image=img1)
    image_label1.place(x=50, y=60)
    lbl3 = Label(new_window, text="Original Image", fg='black', bg='#F5F5F5', font=("Trebuchet MS", 25))
    lbl3.place(x=100, y=5)
    lbl4 = Label(new_window, text="Modified Image", fg='black', bg='skyblue', font=("Trebuchet MS", 25))
    lbl4.place(x=800, y=20)

    operations1 = [
        "Grayscale",
        "Binary",
    ]
    operation_menu = OptionMenu(new_window, operation_var1, *operations1)
    operation_menu.config(width=15, font=("Arial", 12), bg='deep sky blue')
    operation_menu.place(x=25, y=380)

    operations2 = [
        "Salt & Pepper",
        "Gaussian",
        "Poisson",
    ]
    operation_menu = OptionMenu(new_window, operation_var2, *operations2)
    operation_menu.config(width=15, font=("Arial", 12), bg='deep sky blue')
    operation_menu.place(x=300, y=380)

    operations3 = [
        "High pass filter",
        "Median filter",
        "Low pass filter",
        "Butterworth filter",
    ]
    operation_menu = OptionMenu(new_window, operation_var3, *operations3)
    operation_menu.config(width=15, font=("Arial", 12), bg='deep sky blue')
    operation_menu.place(x=150, y=480)
    # apply button
    apply_button = Button(new_window, text='Apply Operation', width=15, bg='steel blue', font=("Arial", 12),
                          command=apply_operation)
    apply_button.place(x=150, y=600)


window.mainloop()

