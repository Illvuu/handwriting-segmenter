"""
Render grayscale ink-alpha stroke images from handwriting-style fonts
and procedural marks.

Output: RGBA PNGs to data/strokes/ (grayscale ink on transparent alpha)

Usage:
    python -m scripts.make_strokes
"""

import math
import os
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import splprep, splev

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "strokes")

# macOS handwriting fonts to try
HANDWRITING_FONTS = [
    "/System/Library/Fonts/Supplemental/Snell Roundhand.ttf",
    "/Library/Fonts/Savoye LET.ttc",
    "/System/Library/Fonts/Supplemental/Apple Chancery.ttf",
    "/Library/Fonts/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Noteworthy.ttc",
    "/System/Library/Fonts/Supplemental/MarkerFelt.ttc",
    "/System/Library/Fonts/Supplemental/Chalkboard SE.ttc",
    "/Library/Fonts/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    # Fallbacks
    "/System/Library/Fonts/Supplemental/Bradley Hand.ttf",
    "/System/Library/Fonts/Supplemental/Noteworthy.ttc",
    "/System/Library/Fonts/Supplemental/Marker Felt.ttc",
    "/Library/Fonts/Comic Sans MS Bold.ttf",
]

# Words/phrases for handwriting samples
WORDS = [
    "Hello", "World", "Test", "Answer", "Name", "Date",
    "Score", "Grade", "Total", "Points", "Yes", "No",
    "octahedral", "non-polar", "tetrahedral", "trigonal",
    "linear", "bent", "polar", "sp3", "sp2", "sp",
    "2x + 3", "y = mx + b", "f(x)", "delta H", "pH = 7",
    "100%", "A+", "3/4", "2.5", "42",
]

FORMULA_TOKENS = [
    "H₂O", "CO₂", "NaCl", "NH₃", "CH₄", "O₂", "N₂",
    "C₆H₁₂O₆", "H₂SO₄", "HCl", "NaOH", "Fe³⁺", "Cu²⁺",
]

SHORT_ANSWERS = [
    "octahedral", "non-polar", "tetrahedral", "trigonal planar",
    "linear", "bent", "polar", "ionic", "covalent", "metallic",
    "exothermic", "endothermic", "equilibrium", "catalyst",
]


def get_available_fonts():
    """Find available handwriting fonts on the system."""
    available = []
    for path in HANDWRITING_FONTS:
        if os.path.exists(path):
            try:
                ImageFont.truetype(path, 40)
                available.append(path)
            except (IOError, OSError):
                continue
    if not available:
        # Use any available system font
        fallbacks = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in fallbacks:
            if os.path.exists(path):
                available.append(path)
                break
    return available


def render_font_stroke(fonts):
    """Render text using a handwriting font as RGBA image."""
    font_path = random.choice(fonts)
    size = random.randint(28, 72)
    try:
        font = ImageFont.truetype(font_path, size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    text = random.choice(WORDS + FORMULA_TOKENS + SHORT_ANSWERS)

    # Measure text
    dummy = Image.new("RGBA", (1, 1))
    dd = ImageDraw.Draw(dummy)
    bbox = dd.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + 20
    th = bbox[3] - bbox[1] + 20

    img = Image.new("RGBA", (max(tw, 10), max(th, 10)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw in grayscale (will be tinted later during compositing)
    ink_value = random.randint(20, 80)
    draw.text((10 - bbox[0], 10 - bbox[1]), text, fill=(ink_value, ink_value, ink_value, 255), font=font)

    return img


def render_bezier_signature():
    """Generate a procedural Bezier signature stroke."""
    w = random.randint(150, 350)
    h = random.randint(40, 100)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Generate control points for a flowing signature
    num_points = random.randint(8, 20)
    points_x = sorted([random.randint(10, w - 10) for _ in range(num_points)])
    points_y = [random.randint(10, h - 10) for _ in range(num_points)]

    try:
        tck, u = splprep([points_x, points_y], s=random.uniform(50, 500), k=min(3, len(points_x) - 1))
        u_new = np.linspace(0, 1, 200)
        x_new, y_new = splev(u_new, tck)

        ink = random.randint(10, 70)
        stroke_width = random.randint(1, 3)
        coords = list(zip(x_new.tolist(), y_new.tolist()))
        for i in range(len(coords) - 1):
            draw.line([coords[i], coords[i + 1]], fill=(ink, ink, ink, 255), width=stroke_width)
    except Exception:
        # Fallback simple line
        draw.line([(10, h // 2), (w - 10, h // 2)], fill=(40, 40, 40, 255), width=2)

    return img


def render_scribble():
    """Generate a random scribble mark."""
    w = random.randint(60, 200)
    h = random.randint(40, 120)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ink = random.randint(15, 70)
    num_strokes = random.randint(3, 10)
    for _ in range(num_strokes):
        x1, y1 = random.randint(5, w - 5), random.randint(5, h - 5)
        x2, y2 = random.randint(5, w - 5), random.randint(5, h - 5)
        draw.line([(x1, y1), (x2, y2)], fill=(ink, ink, ink, 255), width=random.randint(1, 3))

    return img


def render_checkmark():
    """Generate a checkmark."""
    size = random.randint(20, 50)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ink = random.randint(10, 60)
    width = random.randint(2, 4)
    # Checkmark: down-left to bottom-center, then up to top-right
    p1 = (size * 0.1, size * 0.5)
    p2 = (size * 0.35, size * 0.85)
    p3 = (size * 0.9, size * 0.15)
    draw.line([p1, p2], fill=(ink, ink, ink, 255), width=width)
    draw.line([p2, p3], fill=(ink, ink, ink, 255), width=width)

    return img


def render_x_mark():
    """Generate an X mark."""
    size = random.randint(18, 45)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ink = random.randint(10, 60)
    width = random.randint(2, 4)
    margin = size * 0.15
    draw.line([(margin, margin), (size - margin, size - margin)], fill=(ink, ink, ink, 255), width=width)
    draw.line([(size - margin, margin), (margin, size - margin)], fill=(ink, ink, ink, 255), width=width)

    return img


def render_circle_mark():
    """Generate a circle mark."""
    size = random.randint(20, 55)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ink = random.randint(10, 60)
    width = random.randint(1, 3)
    margin = size * 0.1
    draw.ellipse([(margin, margin), (size - margin, size - margin)],
                 outline=(ink, ink, ink, 255), width=width)

    return img


def render_arrow():
    """Generate an arrow mark."""
    w = random.randint(40, 120)
    h = random.randint(20, 50)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ink = random.randint(10, 60)
    width = random.randint(1, 3)
    mid_y = h // 2
    # Shaft
    draw.line([(5, mid_y), (w - 15, mid_y)], fill=(ink, ink, ink, 255), width=width)
    # Arrowhead
    draw.line([(w - 15, mid_y), (w - 25, mid_y - 8)], fill=(ink, ink, ink, 255), width=width)
    draw.line([(w - 15, mid_y), (w - 25, mid_y + 8)], fill=(ink, ink, ink, 255), width=width)

    return img


def render_underline():
    """Generate an underline stroke."""
    w = random.randint(60, 250)
    h = random.randint(10, 25)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ink = random.randint(10, 60)
    width = random.randint(1, 3)
    # Slightly wavy underline
    points = [(x, h // 2 + random.randint(-2, 2)) for x in range(0, w, 5)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(ink, ink, ink, 255), width=width)

    return img


def render_lewis_structure():
    """Generate a maths-structure diagram: central atom, radiating bonds, terminal letters, lone-pair dots, dipole arrows."""
    size = random.randint(100, 200)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ink = random.randint(15, 65)
    color = (ink, ink, ink, 255)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", random.randint(14, 22))
    except (IOError, OSError):
        font = ImageFont.load_default()

    center = (size // 2, size // 2)
    central_atom = random.choice(["C", "N", "O", "S", "P", "Si", "B"])
    draw.text((center[0] - 6, center[1] - 8), central_atom, fill=color, font=font)

    # Radiating bonds
    terminal_atoms = random.choices(["H", "F", "Cl", "O", "N", "CH₃"], k=random.randint(2, 4))
    bond_length = size // 3
    num_bonds = len(terminal_atoms)

    for i, atom in enumerate(terminal_atoms):
        angle = (2 * math.pi * i) / num_bonds + random.uniform(-0.2, 0.2)
        ex = center[0] + int(bond_length * math.cos(angle))
        ey = center[1] + int(bond_length * math.sin(angle))

        # Bond line (single, double, or triple)
        bond_type = random.choice([1, 1, 1, 2, 3])
        for b in range(bond_type):
            offset = (b - bond_type // 2) * 3
            draw.line([(center[0] + offset, center[1]), (ex + offset, ey)],
                      fill=color, width=1)

        # Terminal atom
        draw.text((ex - 5, ey - 8), atom, fill=color, font=font)

    # Lone-pair dots
    num_lone_pairs = random.randint(0, 3)
    for _ in range(num_lone_pairs):
        angle = random.uniform(0, 2 * math.pi)
        dist = bond_length + random.randint(5, 15)
        dx = center[0] + int(dist * math.cos(angle))
        dy = center[1] + int(dist * math.sin(angle))
        draw.ellipse([dx - 2, dy - 2, dx + 2, dy + 2], fill=color)
        draw.ellipse([dx + 4, dy - 2, dx + 8, dy + 2], fill=color)

    # Optional dipole arrow
    if random.random() > 0.5:
        arrow_y = size - 15
        draw.line([(10, arrow_y), (size - 20, arrow_y)], fill=color, width=1)
        draw.line([(size - 20, arrow_y), (size - 28, arrow_y - 4)], fill=color, width=1)
        draw.line([(size - 20, arrow_y), (size - 28, arrow_y + 4)], fill=color, width=1)
        # Plus sign at tail
        draw.text((5, arrow_y - 12), "+", fill=color, font=font)

    return img


def render_formula_with_scripts():
    """Render handwritten formula tokens with sub/superscripts."""
    try:
        fonts = get_available_fonts()
        font_path = random.choice(fonts) if fonts else None
        main_size = random.randint(24, 48)
        sub_size = int(main_size * 0.6)
        if font_path:
            main_font = ImageFont.truetype(font_path, main_size)
            sub_font = ImageFont.truetype(font_path, sub_size)
        else:
            main_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()
    except (IOError, OSError):
        main_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    # Build formula
    elements = random.choices(["H", "C", "O", "N", "S", "Fe", "Ca", "Na", "K"], k=random.randint(2, 4))
    subscripts = [str(random.randint(1, 12)) if random.random() > 0.3 else "" for _ in elements]

    w = random.randint(120, 250)
    h = random.randint(50, 80)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ink = random.randint(15, 70)
    color = (ink, ink, ink, 255)
    x = 10
    baseline = h // 3

    for elem, sub in zip(elements, subscripts):
        draw.text((x, baseline), elem, fill=color, font=main_font)
        x += main_size * 0.6
        if sub:
            draw.text((x, baseline + main_size * 0.4), sub, fill=color, font=sub_font)
            x += sub_size * 0.5
        x += random.randint(0, 5)

    # Optional superscript (charge)
    if random.random() > 0.5:
        charge = random.choice(["²⁺", "³⁺", "⁻", "²⁻", "+", "-"])
        draw.text((x, baseline - main_size * 0.3), charge, fill=color, font=sub_font)

    return img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fonts = get_available_fonts()
    if not fonts:
        print("WARNING: No handwriting fonts found, using fallback")

    # Target: ~3000-5000 stroke images
    num_font_strokes = 1500
    num_procedural = 2000

    generators_procedural = [
        (render_bezier_signature, 300),
        (render_scribble, 250),
        (render_checkmark, 250),
        (render_x_mark, 200),
        (render_circle_mark, 200),
        (render_arrow, 150),
        (render_underline, 200),
        (render_lewis_structure, 250),
        (render_formula_with_scripts, 200),
    ]

    idx = 0

    # Font-based strokes
    print(f"Generating {num_font_strokes} font-based strokes...")
    for i in range(num_font_strokes):
        if fonts:
            img = render_font_stroke(fonts)
        else:
            img = render_scribble()

        path = os.path.join(OUTPUT_DIR, f"stroke_{idx:05d}.png")
        img.save(path, "PNG")
        idx += 1
        if (i + 1) % 500 == 0:
            print(f"  Font strokes: {i + 1}/{num_font_strokes}")

    # Procedural marks
    print(f"Generating procedural marks...")
    for gen_func, count in generators_procedural:
        for i in range(count):
            img = gen_func()
            path = os.path.join(OUTPUT_DIR, f"stroke_{idx:05d}.png")
            img.save(path, "PNG")
            idx += 1

    # Short answer words (special category)
    print("Generating short answer words...")
    for word in SHORT_ANSWERS * 5:
        if fonts:
            font_path = random.choice(fonts)
            size = random.randint(24, 56)
            try:
                font = ImageFont.truetype(font_path, size)
            except (IOError, OSError):
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()
            size = 16

        dummy = Image.new("RGBA", (1, 1))
        dd = ImageDraw.Draw(dummy)
        bbox = dd.textbbox((0, 0), word, font=font)
        tw = bbox[2] - bbox[0] + 20
        th = bbox[3] - bbox[1] + 20

        img = Image.new("RGBA", (max(tw, 10), max(th, 10)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        ink = random.randint(15, 70)
        draw.text((10 - bbox[0], 10 - bbox[1]), word, fill=(ink, ink, ink, 255), font=font)

        path = os.path.join(OUTPUT_DIR, f"stroke_{idx:05d}.png")
        img.save(path, "PNG")
        idx += 1

    print(f"Done. Generated {idx} stroke images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
