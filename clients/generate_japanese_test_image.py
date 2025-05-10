from PIL import Image, ImageDraw, ImageFont
import os

# --- Configuration ---
IMG_WIDTH = 900
IMG_HEIGHT = 600
BACKGROUND_COLOR = (250, 250, 245) # Light beige background
OUTPUT_FILENAME_BASE = "japanese_ocr_test_"
FONT_SIZE_LARGE = 48
FONT_SIZE_MEDIUM = 36
FONT_SIZE_SMALL = 24
TEXT_COLOR_PRIMARY = (30, 30, 30)   # Dark Gray
TEXT_COLOR_SECONDARY = (10, 10, 90) # Dark Blue

# --- Potential Japanese Font Paths/Names ---
# Add paths relevant to YOUR system if needed. Case might matter on Linux.
# Using Noto Sans CJK JP as a common example for Linux.
# Using Meiryo for Windows, Hiragino for macOS.
POTENTIAL_JP_FONTS = [
    # Linux Examples (adjust paths based on installation)
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', # Noto Sans CJK (might need index)
    '/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf', # Specific Noto JP
    '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf',
    '/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf',
    # Windows Examples
    'C:/Windows/Fonts/meiryo.ttc',
    'C:/Windows/Fonts/msgothic.ttc',
    'C:/Windows/Fonts/yugothr.ttf',
    # macOS Examples
    '/System/Library/Fonts/Hiragino Sans GB.ttc', # Check exact Hiragino name
    '/Library/Fonts/ヒラギノ角ゴシック W3.ttc', # Example name, check Font Book
    '/Library/Fonts/Hiragino Sans W3.otf',
    # Provide a direct path if known:
    # '/path/to/your/japanese_font.ttf'
]

# --- Text Content (Japanese) ---
japanese_texts = [
    {"text": "こんにちは世界", "size": FONT_SIZE_LARGE, "color": TEXT_COLOR_PRIMARY, "pos": (50, 30)},
    {"text": "これは日本語のOCRテストです。", "size": FONT_SIZE_MEDIUM, "color": TEXT_COLOR_SECONDARY, "pos": (50, 100)},
    {"text": "カタカナとひらがな", "size": FONT_SIZE_MEDIUM, "color": TEXT_COLOR_PRIMARY, "pos": (50, 160)},
    {"text": "テキスト認識", "size": FONT_SIZE_LARGE, "color": TEXT_COLOR_SECONDARY, "pos": (50, 220)},
    {"text": "背景からテキストを抽出", "size": FONT_SIZE_MEDIUM, "color": TEXT_COLOR_PRIMARY, "pos": (50, 290)},
    {"text": "一二三四五六七八九十", "size": FONT_SIZE_MEDIUM, "color": TEXT_COLOR_SECONDARY, "pos": (50, 350)},
    {"text": "株式会社PaddlePaddle", "size": FONT_SIZE_SMALL, "color": TEXT_COLOR_PRIMARY, "pos": (50, 410)},
    {"text": "東京都", "size": FONT_SIZE_LARGE, "color": TEXT_COLOR_SECONDARY, "pos": (50, 460)},
    {"text": "ありがとうございます", "size": FONT_SIZE_MEDIUM, "color": TEXT_COLOR_PRIMARY, "pos": (50, 530)},
]

# Store loaded fonts
loaded_fonts = {}
selected_font_path = None # Store the path of the first successfully loaded font
input_dir = "inputs"

def get_font(size):
    """ Attempts to load a Japanese font from potential paths. """
    global selected_font_path
    font_key = (selected_font_path, size) # Use selected path once found

    if font_key in loaded_fonts:
        return loaded_fonts[font_key]

    # If we haven't found a working font yet, search the list
    if selected_font_path is None:
        for path in POTENTIAL_JP_FONTS:
            try:
                # .ttc files might require specifying the index (try 0)
                font = ImageFont.truetype(path, size, index=0)
                print(f"Successfully loaded Japanese font: {path}")
                selected_font_path = path # Remember the working font path
                font_key = (selected_font_path, size) # Update key
                loaded_fonts[font_key] = font
                return font
            except IOError:
                continue # Font not found at this path or invalid
            except Exception as e: # Catch other potential errors like index errors for non-TTC
                print(f"Warning: Could not load font '{path}' (Size: {size}). Error: {e}")
                continue

    # If we already selected a font, try loading it with the new size
    elif selected_font_path:
         try:
             font = ImageFont.truetype(selected_font_path, size, index=0)
             loaded_fonts[font_key] = font
             return font
         except Exception as e:
             print(f"Error: Could not reload font '{selected_font_path}' with size {size}. Error: {e}")
             # Reset selected_font_path to None to trigger search next time if needed
             selected_font_path = None
             return None

    # If no font was found after searching
    print("-" * 60)
    print("ERROR: Could not load any Japanese font from the potential paths:")
    for p in POTENTIAL_JP_FONTS: print(f"  - {p}")
    print("Please ensure a Japanese TTF/OTF font is installed and accessible,")
    print("or add the correct path to the POTENTIAL_JP_FONTS list in the script.")
    print("-" * 60)
    return None

# --- Create and Save Image ---
def create_image(text_list, index):
    global selected_font_path
    selected_font_path = None # Reset selected font for each image generation if needed
    loaded_fonts.clear()

    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    print(f"\nGenerating image {index}...")

    font_error = False
    for element in text_list:
        font_obj = get_font(element["size"])
        if font_obj:
            try:
                # Use textbbox for potentially better positioning with complex fonts
                # box = draw.textbbox(element["pos"], element["text"], font=font_obj)
                # draw.text(box[:2], element["text"], fill=element["color"], font=font_obj)
                # Simple text drawing:
                draw.text(element["pos"], element["text"], fill=element["color"], font=font_obj)
            except Exception as e:
                print(f"Error drawing text '{element['text']}': {e}")
        else:
            font_error = True # Flag that we couldn't load the font

    if font_error:
        print(f"WARNING: Image {index} generated, but some text could not be rendered due to missing font.")
    else:
        print(f"Finished drawing elements for image {index}.")

    # Save the Image
    filename = f"{input_dir}/{OUTPUT_FILENAME_BASE}{index+1}.png"
    img.save(filename)
    print(f"Test image saved successfully as '{filename}'")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # You can create multiple images if needed, e.g., by splitting japanese_texts
        create_image(japanese_texts, 2)
        # create_image(other_japanese_text_list, 1) # Example for a second image

    except ImportError:
         print("Error: Pillow library not found.")
         print("Please install it using: pip install Pillow")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")