"""
Generate synthetic form pages (templates) at 150 DPI.
Produces 1275-1650 template images with diverse form elements.
Each template emits a .cells.json sidecar listing empty-cell rectangles.

Usage:
    python -m scripts.make_templates
"""

import json
import math
import os
import random
import string
import sys

from PIL import Image, ImageDraw, ImageFont

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "templates")

# Page dimensions at 150 DPI (letter size 8.5 x 11 inches)
DPI = 150
PAGE_W = int(8.5 * DPI)  # 1275
PAGE_H = int(11 * DPI)   # 1650

# Font sizes
TITLE_SIZE = 24
HEADER_SIZE = 18
BODY_SIZE = 14
SMALL_SIZE = 12
DENSE_SIZE = 13


def get_font(size, bold=False):
    """Get a font at the specified size, with fallbacks."""
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    if bold:
        font_candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
        ] + font_candidates

    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=0)
            except (IOError, OSError):
                continue
    return ImageFont.load_default()


def draw_field_rows(draw, y, cells, page_w):
    """Draw labeled input field rows (name, date, etc.)."""
    labels = random.sample([
        "Name", "Date", "ID Number", "Address", "City", "State", "Zip",
        "Phone", "Email", "Employer", "SSN", "DOB", "Grade", "Section",
        "Period", "Teacher", "School", "Course", "Subject"
    ], k=random.randint(3, 7))

    font = get_font(BODY_SIZE)
    for label in labels:
        x = random.randint(40, 100)
        draw.text((x, y + 4), f"{label}:", fill="black", font=font)
        # Field line
        field_x = x + 120
        field_w = random.randint(200, page_w - field_x - 60)
        draw.line([(field_x, y + 20), (field_x + field_w, y + 20)], fill="black", width=1)
        cells.append([field_x, y, field_w, 24])
        y += random.randint(30, 45)
    return y


def draw_numbered_boxes(draw, y, cells, page_w):
    """Draw shaded numbered boxes (like bubble-sheet or answer boxes)."""
    num_boxes = random.randint(10, 30)
    box_size = random.randint(28, 40)
    spacing = box_size + random.randint(4, 10)
    cols = min(num_boxes, (page_w - 100) // spacing)
    rows = math.ceil(num_boxes / cols)

    x_start = random.randint(50, 100)
    font = get_font(SMALL_SIZE)

    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if idx >= num_boxes:
                break
            bx = x_start + col * spacing
            by = y + row * spacing
            # Shaded background
            shade = random.randint(220, 240)
            draw.rectangle([bx, by, bx + box_size, by + box_size],
                           fill=(shade, shade, shade), outline="black")
            # Number
            draw.text((bx + 3, by + 2), str(idx + 1), fill="black", font=font)
            cells.append([bx, by, box_size, box_size])

    return y + rows * spacing + 20


def draw_table(draw, y, cells, page_w):
    """Draw a table with various column widths."""
    num_cols = random.randint(3, 7)
    num_rows = random.randint(3, 8)
    x_start = random.randint(40, 80)
    available_w = page_w - x_start - 40

    # Random column widths
    col_weights = [random.uniform(0.5, 2.0) for _ in range(num_cols)]
    total_weight = sum(col_weights)
    col_widths = [int(available_w * w / total_weight) for w in col_weights]

    row_height = random.randint(24, 40)
    font = get_font(SMALL_SIZE)

    # Header row
    header_labels = random.sample([
        "Item", "Qty", "Price", "Total", "Date", "Description",
        "Score", "Grade", "Points", "Answer", "Value", "Weight"
    ], k=min(num_cols, 12))

    for row in range(num_rows + 1):
        cx = x_start
        for col in range(num_cols):
            rx = cx
            ry = y + row * row_height
            rw = col_widths[col]
            rh = row_height
            draw.rectangle([rx, ry, rx + rw, ry + rh], outline="black")
            if row == 0 and col < len(header_labels):
                draw.text((rx + 4, ry + 4), header_labels[col], fill="black", font=font)
            else:
                cells.append([rx + 1, ry + 1, rw - 2, rh - 2])
            cx += col_widths[col]

    return y + (num_rows + 1) * row_height + 15


def draw_checkboxes(draw, y, cells, page_w):
    """Draw checkbox rows (hollow squares)."""
    num_items = random.randint(4, 10)
    box_size = random.randint(14, 20)
    font = get_font(BODY_SIZE)

    options = [
        "Agree", "Disagree", "True", "False", "Yes", "No",
        "Option A", "Option B", "Option C", "Option D",
        "Strongly Agree", "Neutral", "N/A", "Other"
    ]

    for i in range(num_items):
        x = random.choice([60, 80, page_w // 2 + 40])
        draw.rectangle([x, y, x + box_size, y + box_size], outline="black", width=1)
        cells.append([x, y, box_size, box_size])
        label = random.choice(options)
        draw.text((x + box_size + 8, y + 1), label, fill="black", font=font)
        y += box_size + random.randint(8, 16)

    return y


def draw_signature_block(draw, y, cells, page_w):
    """Draw signature blocks with lines and labels."""
    font = get_font(SMALL_SIZE)
    blocks = random.randint(1, 3)

    for _ in range(blocks):
        x = random.randint(50, 150)
        line_w = random.randint(250, 450)
        draw.line([(x, y + 30), (x + line_w, y + 30)], fill="black", width=1)
        label = random.choice(["Signature", "Date", "Print Name", "Witness", "Parent/Guardian"])
        draw.text((x, y + 34), label, fill="black", font=font)
        cells.append([x, y, line_w, 30])
        y += 55

    return y


def draw_answer_table_with_work_cells(draw, y, cells, page_w):
    """Draw answer table with tall work cells, some with shaded inner answer boxes."""
    num_problems = random.randint(4, 10)
    x_start = random.randint(40, 80)
    work_height = random.randint(80, 140)
    answer_box_h = random.randint(28, 40)
    col_w = min((page_w - x_start - 40) // min(num_problems, 5), 200)
    cols = min(num_problems, 5)
    rows = math.ceil(num_problems / cols)

    font = get_font(SMALL_SIZE)

    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if idx >= num_problems:
                break
            cx = x_start + col * col_w
            cy = y + row * (work_height + 10)

            # Main work cell
            draw.rectangle([cx, cy, cx + col_w, cy + work_height], outline="black")
            draw.text((cx + 4, cy + 2), str(idx + 1) + ".", fill="black", font=font)
            cells.append([cx + 1, cy + 1, col_w - 2, work_height - 2])

            # Shaded answer box inside (random chance)
            if random.random() > 0.4:
                ax = cx + col_w - answer_box_h - 8
                ay = cy + work_height - answer_box_h - 8
                shade = random.randint(215, 235)
                draw.rectangle([ax, ay, ax + answer_box_h, ay + answer_box_h],
                               fill=(shade, shade, shade), outline="black")

    return y + rows * (work_height + 10) + 15


def draw_chemical_notation_lines(draw, y, cells, page_w):
    """Draw printed chemical/notation lines with subscripted formula tokens and dot-structure rows."""
    font = get_font(BODY_SIZE)
    small_font = get_font(10)

    formulas = [
        "H₂O", "CO₂", "NaCl", "C₆H₁₂O₆", "NH₃", "H₂SO₄",
        "CH₄", "C₂H₅OH", "Fe₂O₃", "Ca(OH)₂", "KMnO₄", "HNO₃"
    ]

    num_lines = random.randint(3, 7)
    for _ in range(num_lines):
        x = random.randint(50, 120)
        formula = random.choice(formulas)
        draw.text((x, y), formula, fill="black", font=font)

        # Dot structure representation (dots around atom)
        dot_x = x + 150 + random.randint(0, 100)
        atom = random.choice(["C", "N", "O", "S", "P", "F", "Cl"])
        draw.text((dot_x, y), atom, fill="black", font=font)

        # Draw dots around atom
        for dx, dy in [(-8, 0), (12, 0), (2, -10), (2, 14), (-8, -10), (12, 14)]:
            if random.random() > 0.4:
                draw.ellipse([dot_x + dx, y + dy + 4, dot_x + dx + 3, y + dy + 7], fill="black")

        y += random.randint(28, 40)

    return y


def draw_inverted_header(draw, y, cells, page_w):
    """Draw inverted white-on-black section headers ('Part I'-style bars)."""
    font = get_font(HEADER_SIZE, bold=True)
    parts = ["Part I", "Part II", "Part III", "Part IV",
             "Section A", "Section B", "Section C",
             "ANSWER KEY", "INSTRUCTIONS", "SCORING"]
    label = random.choice(parts)

    bar_h = random.randint(26, 36)
    x_start = random.randint(30, 60)
    bar_w = page_w - x_start * 2

    draw.rectangle([x_start, y, x_start + bar_w, y + bar_h], fill="black")
    draw.text((x_start + 10, y + 4), label, fill="white", font=font)

    return y + bar_h + 10


def draw_lettered_lines_with_checkboxes(draw, y, cells, page_w):
    """Draw lettered lines with dotted leaders ending in small hollow checkboxes."""
    font = get_font(BODY_SIZE)
    num_items = random.randint(5, 12)

    for i in range(num_items):
        letter = chr(ord('a') + i) if i < 26 else str(i)
        x = random.randint(50, 80)
        draw.text((x, y), f"{letter}.", fill="black", font=font)

        # Dotted leader
        leader_start = x + 25
        leader_end = page_w - 80
        dot_spacing = 8
        for dx in range(leader_start, leader_end, dot_spacing):
            draw.ellipse([dx, y + 10, dx + 2, y + 12], fill="black")

        # Small checkbox at end
        cb_size = 12
        cb_x = leader_end + 10
        draw.rectangle([cb_x, y + 2, cb_x + cb_size, y + 2 + cb_size], outline="black", width=1)
        cells.append([cb_x, y + 2, cb_size, cb_size])

        y += random.randint(22, 30)

    return y


def draw_corner_label_cells(draw, y, cells, page_w):
    """Draw bordered cells with printed corner labels ('2c Employer's telephone number' style)."""
    font = get_font(10)
    labels = [
        "1 Wages, tips, other compensation",
        "2 Federal income tax withheld",
        "3 Social security wages",
        "4 Social security tax withheld",
        "5 Medicare wages and tips",
        "6 Medicare tax withheld",
        "2c Employer's telephone number",
        "7 Social security tips",
        "8 Allocated tips",
        "9 Verification code"
    ]

    num_cells = random.randint(4, 8)
    cell_w = random.randint(180, 280)
    cell_h = random.randint(45, 65)
    cols = min(num_cells, (page_w - 80) // (cell_w + 5))
    rows = math.ceil(num_cells / cols)
    x_start = random.randint(40, 60)

    chosen_labels = random.sample(labels, k=min(num_cells, len(labels)))

    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= num_cells:
                break
            cx = x_start + col * (cell_w + 4)
            cy = y + row * (cell_h + 4)
            draw.rectangle([cx, cy, cx + cell_w, cy + cell_h], outline="black")
            draw.text((cx + 3, cy + 2), chosen_labels[idx], fill="black", font=font)
            cells.append([cx + 1, cy + 14, cell_w - 2, cell_h - 15])
            idx += 1

    return y + rows * (cell_h + 4) + 15


def draw_bold_year(draw, y, cells, page_w):
    """Draw bold standalone years (38-54 px)."""
    size = random.randint(38, 54)
    font = get_font(size, bold=True)
    year = str(random.randint(2020, 2026))
    x = random.randint(page_w // 3, page_w * 2 // 3)
    draw.text((x, y), year, fill="black", font=font)
    return y + size + 15


def draw_dense_small_print(draw, y, cells, page_w):
    """Draw dense small-print text (12-15 px fonts) covering most of page."""
    size = random.randint(12, 15)
    font = get_font(size)

    paragraphs = [
        "The information contained herein is confidential and proprietary.",
        "All responses must be completed in ink. Do not use correction fluid.",
        "By signing below you certify that all information provided is accurate.",
        "Failure to comply with instructions may result in disqualification.",
        "This form must be returned within 30 days of the date shown above.",
        "Please print clearly using block letters in the spaces provided.",
        "Any alterations or erasures must be initialed by the applicant.",
        "Refer to the instruction booklet for additional guidance on completion.",
        "The department reserves the right to verify all information submitted.",
        "Under penalties of perjury, I declare that I have examined this return.",
        "Include all income from wages, salaries, tips, and other compensation.",
        "Attach all supporting documentation as specified in the instructions.",
        "Round all amounts to the nearest dollar. Do not include cents.",
        "If filing jointly, both signatures are required below.",
        "Estimated tax payments should be made quarterly using form ES-1040.",
    ]

    x_margin = random.randint(40, 70)
    line_height = size + random.randint(3, 6)
    max_lines = random.randint(30, 50)

    for i in range(max_lines):
        text = random.choice(paragraphs)
        if random.random() > 0.7:
            text = text.upper()
        draw.text((x_margin, y), text, fill="black", font=font)
        y += line_height
        if y > PAGE_H - 80:
            break

    return y


def generate_template(idx):
    """Generate a single template page with random form elements."""
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)
    cells = []

    # Title
    title_font = get_font(TITLE_SIZE, bold=True)
    title = f"SAMPLE {''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 8)))} FORM {random.randint(100, 9999)}"
    draw.text((PAGE_W // 2 - 150, 20), title, fill="black", font=title_font)

    y = 60

    # Decide if this is a dense-small-print page (~30+ out of total)
    is_dense = idx < 35

    if is_dense:
        y = draw_dense_small_print(draw, y, cells, PAGE_W)
    else:
        # Randomly select and draw various sections
        section_funcs = [
            draw_field_rows,
            draw_numbered_boxes,
            draw_table,
            draw_checkboxes,
            draw_signature_block,
            draw_answer_table_with_work_cells,
            draw_chemical_notation_lines,
            draw_inverted_header,
            draw_lettered_lines_with_checkboxes,
            draw_corner_label_cells,
            draw_bold_year,
        ]

        # Pick 3-7 sections per page
        num_sections = random.randint(3, 7)
        chosen = random.sample(section_funcs, k=min(num_sections, len(section_funcs)))

        for func in chosen:
            if y > PAGE_H - 150:
                break
            try:
                if func in (draw_inverted_header, draw_bold_year):
                    y = func(draw, y, cells, PAGE_W)
                else:
                    y = func(draw, y, cells, PAGE_W)
            except Exception:
                pass
            y += random.randint(10, 30)

    return img, cells


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    num_templates = random.randint(1275, 1650)
    print(f"Generating {num_templates} templates to {OUTPUT_DIR}")

    for i in range(num_templates):
        img, cells = generate_template(i)

        img_path = os.path.join(OUTPUT_DIR, f"template_{i:04d}.png")
        json_path = os.path.join(OUTPUT_DIR, f"template_{i:04d}.cells.json")

        img.save(img_path, "PNG")
        with open(json_path, "w") as f:
            json.dump(cells, f)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_templates}")

    print(f"Done. {num_templates} templates generated.")


if __name__ == "__main__":
    main()
