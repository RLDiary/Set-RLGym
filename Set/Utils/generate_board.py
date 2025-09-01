import cairo
import math
import itertools
from typing import List, Tuple, Optional
import io
from PIL import Image

# Colors (RGB values 0-1 for Cairo)
COLORS = {
    'green': (1.0, 0.757, 0.027),
    'blue': (0.118, 0.533, 0.898), 
    'red': (0.847, 0.106, 0.376)
}

# Map from env.py color indices to Cairo color names
COLOR_MAP = {0: 'red', 1: 'green', 2: 'blue'}
SHADING_MAP = {0: 'solid', 1: 'striped', 2: 'empty'}
SHAPE_MAP = {0: 'diamond', 1: 'squiggle', 2: 'oval'}

def _draw_diamond(ctx, cx, cy, size, color, shading, scale_factor):
    """Draw diamond shape with proper shading"""
    ctx.save()
    
    # Create diamond path with wider horizontal span
    horizontal_size = size * 1.4  # Extend left and right points further
    ctx.move_to(cx, cy - size)
    ctx.line_to(cx + horizontal_size, cy)
    ctx.line_to(cx, cy + size)
    ctx.line_to(cx - horizontal_size, cy)
    ctx.close_path()
    
    if shading == 'solid':
        ctx.set_source_rgb(*color)
        ctx.fill_preserve()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(0.5 * scale_factor)
        ctx.stroke()
    elif shading == 'striped':
        # Fill with white background
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        
        # Save the path for clipping
        ctx.clip()
        
        # Draw denser stripes
        ctx.set_source_rgb(*color)
        ctx.set_line_width(0.5 * scale_factor)
        stripe_spacing = max(1, int(2 * scale_factor))
        for i in range(-int(horizontal_size*2), int(horizontal_size*2), stripe_spacing):
            ctx.move_to(cx + i, cy - size*2)
            ctx.line_to(cx + i, cy + size*2)
            ctx.stroke()
        
        ctx.restore()
        ctx.save()
        
        # Redraw outline
        ctx.move_to(cx, cy - size)
        ctx.line_to(cx + horizontal_size, cy)
        ctx.line_to(cx, cy + size)
        ctx.line_to(cx - horizontal_size, cy)
        ctx.close_path()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1 * scale_factor)
        ctx.stroke()
    else:  # empty
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1 * scale_factor)
        ctx.stroke()
    
    ctx.restore()

def _draw_oval(ctx, cx, cy, width, height, color, shading, scale_factor):
    """Draw oval shape with proper shading"""
    ctx.save()
    
    # Create oval path
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(width, height)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    
    if shading == 'solid':
        ctx.set_source_rgb(*color)
        ctx.fill_preserve()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(0.5 * scale_factor)
        ctx.stroke()
    elif shading == 'striped':
        # Fill with white background
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        
        # Save the path for clipping
        ctx.clip()
        
        # Draw denser stripes
        ctx.set_source_rgb(*color)
        ctx.set_line_width(0.5 * scale_factor)
        stripe_spacing = max(1, int(2 * scale_factor))
        for i in range(-int(width), int(width), stripe_spacing):
            ctx.move_to(cx + i, cy - height*2)
            ctx.line_to(cx + i, cy + height*2)
            ctx.stroke()
        
        ctx.restore()
        ctx.save()
        
        # Redraw outline
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(width, height)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1 * scale_factor)
        ctx.stroke()
    else:  # empty
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1 * scale_factor)
        ctx.stroke()
    
    ctx.restore()

def _draw_squiggle(ctx, cx, cy, size, color, shading, scale_factor):
    """Draw squiggle/tilde shape as closed path with well-separated top and bottom curves"""
    ctx.save()
    
    # Much larger separation between curves
    thickness = size * 1.2
    
    # Create closed tilde path with better separation
    # Start at bottom-left of bottom curve
    ctx.move_to(cx - size, cy + thickness/2)
    
    # Bottom curve (left to right) - inverted wave pattern
    ctx.curve_to(cx - size*0.6, cy + thickness/2 - size*0.5,
                 cx - size*0.1, cy + thickness/2 - size*0.3,
                 cx + size*0.1, cy + thickness/2 + size*0.1)
    ctx.curve_to(cx + size*0.3, cy + thickness/2 + size*0.4,
                 cx + size*0.7, cy + thickness/2 + size*0.6,
                 cx + size, cy + thickness/2 + size*0.2)
    
    # Right edge connecting bottom to top
    ctx.line_to(cx + size, cy - thickness/2 + size*0.2)
    
    # Top curve (right to left) - mirror of bottom but offset
    ctx.curve_to(cx + size*0.7, cy - thickness/2 + size*0.6,
                 cx + size*0.3, cy - thickness/2 + size*0.4,
                 cx + size*0.1, cy - thickness/2 - size*0.1)
    ctx.curve_to(cx - size*0.1, cy - thickness/2 - size*0.3,
                 cx - size*0.6, cy - thickness/2 - size*0.5,
                 cx - size, cy - thickness/2)
    
    # Left edge - close the path
    ctx.close_path()
    
    if shading == 'solid':
        ctx.set_source_rgb(*color)
        ctx.fill_preserve()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(0.5 * scale_factor)
        ctx.stroke()
    elif shading == 'striped':
        # Fill with white background
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        
        # Save the path for clipping
        ctx.clip()
        
        # Draw denser vertical stripes
        ctx.set_source_rgb(*color)
        ctx.set_line_width(0.5 * scale_factor)
        stripe_spacing = max(1, int(2 * scale_factor))
        for i in range(-int(size*2), int(size*2), stripe_spacing):
            ctx.move_to(cx + i, cy - size)
            ctx.line_to(cx + i, cy + size)
            ctx.stroke()
        
        ctx.restore()
        ctx.save()
        
        # Redraw outline with thicker line
        ctx.move_to(cx - size, cy + thickness/2)
        ctx.curve_to(cx - size*0.6, cy + thickness/2 - size*0.5,
                     cx - size*0.1, cy + thickness/2 - size*0.3,
                     cx + size*0.1, cy + thickness/2 + size*0.1)
        ctx.curve_to(cx + size*0.3, cy + thickness/2 + size*0.4,
                     cx + size*0.7, cy + thickness/2 + size*0.6,
                     cx + size, cy + thickness/2 + size*0.2)
        ctx.line_to(cx + size, cy - thickness/2 + size*0.2)
        ctx.curve_to(cx + size*0.7, cy - thickness/2 + size*0.6,
                     cx + size*0.3, cy - thickness/2 + size*0.4,
                     cx + size*0.1, cy - thickness/2 - size*0.1)
        ctx.curve_to(cx - size*0.1, cy - thickness/2 - size*0.3,
                     cx - size*0.6, cy - thickness/2 - size*0.5,
                     cx - size, cy - thickness/2)
        ctx.close_path()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1 * scale_factor)
        ctx.stroke()
    else:  # empty
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill_preserve()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1 * scale_factor)
        ctx.stroke()
    
    ctx.restore()

def _draw_card(ctx, card_x, card_y, shape, color, shading, number, card_width, card_height, scale_factor, card_number=None):
    """Draw a complete SET card"""
    ctx.save()
    
    # Card background
    ctx.rectangle(card_x, card_y, card_width, card_height)
    ctx.set_source_rgb(1, 1, 1)  # white
    ctx.fill_preserve()
    ctx.set_source_rgb(0.8, 0.8, 0.8)  # light gray border
    ctx.set_line_width(0.5 * scale_factor)
    ctx.stroke()
    
    # Symbol positioning - increased size for better visibility
    symbol_size = int(30 * scale_factor)  # Increased from 20 to 30
    card_center_x = card_x + card_width // 2
    card_center_y = card_y + card_height // 2
    
    if number == 1:
        positions = [(card_center_x, card_center_y)]
    elif number == 2:
        offset = int(18 * scale_factor)
        positions = [
            (card_center_x, card_center_y - offset),
            (card_center_x, card_center_y + offset)
        ]
    else:  # number == 3
        offset = int(40 * scale_factor)  # Increased from 25 to 40 to accommodate larger shapes
        positions = [
            (card_center_x, card_center_y - offset),
            (card_center_x, card_center_y),
            (card_center_x, card_center_y + offset)
        ]
    
    # Draw symbols
    for pos_x, pos_y in positions:
        if shape == 'diamond':
            _draw_diamond(ctx, pos_x, pos_y, symbol_size//2, COLORS[color], shading, scale_factor)
        elif shape == 'oval':
            _draw_oval(ctx, pos_x, pos_y, symbol_size*0.6, symbol_size*0.4, COLORS[color], shading, scale_factor)
        elif shape == 'squiggle':
            _draw_squiggle(ctx, pos_x, pos_y, symbol_size*0.6, COLORS[color], shading, scale_factor)
    
    # Draw card number in black circle at top left
    if card_number is not None:
        circle_radius = 15 * scale_factor
        circle_x = card_x + circle_radius + 8
        circle_y = card_y + circle_radius + 8
        
        # Draw black circle
        ctx.arc(circle_x, circle_y, circle_radius, 0, 2 * math.pi)
        ctx.set_source_rgb(0, 0, 0)  # black
        ctx.fill()
        
        # Draw white number text
        ctx.set_source_rgb(1, 1, 1)  # white text
        font_size = min(20 * scale_factor, circle_radius * 1.2)
        ctx.set_font_size(font_size)
        
        text = str(card_number)
        text_extents = ctx.text_extents(text)
        text_x = circle_x - text_extents.width / 2
        text_y = circle_y + text_extents.height / 2
        
        ctx.move_to(text_x, text_y)
        ctx.show_text(text)
    
    ctx.restore()

def code_to_cairo_params(code: int) -> Tuple[str, str, str, int]:
    """Convert env.py card code to Cairo parameters.
    Returns (shape, color, shading, number) compatible with Cairo functions.
    """
    # Decode using env.py's scheme: NUMBER, COLOR, SHADING, SHAPE
    n = code % 3  # number: 0,1,2 -> 1,2,3
    c = (code // 3) % 3  # color index
    s = (code // 9) % 3  # shading index
    h = (code // 27) % 3  # shape index
    
    shape = SHAPE_MAP[h]
    color = COLOR_MAP[c]
    shading = SHADING_MAP[s]
    number = n + 1  # Convert 0,1,2 to 1,2,3
    
    return shape, color, shading, number

def generate_board_image(codes: List[int], 
                        card_width: int = 220, 
                        card_height: int = 320, 
                        margin: int = 24,
                        cols: int = 4,
                        rows: int = 3,
                        overlay_indices: bool = True) -> Image.Image:
    """Generate a board image from a list of card codes using Cairo graphics.
    
    Args:
        codes: List of card codes from env.py encoding
        card_width: Width of each card in pixels
        card_height: Height of each card in pixels  
        margin: Margin between cards in pixels
        cols: Number of columns in the grid
        rows: Number of rows in the grid (3 cards per column)
        overlay_indices: Whether to show card index numbers
        
    Returns:
        PIL Image of the rendered board
    """
    # Use provided rows and cols (calculated based on 3 cards per column)
    
    # Calculate canvas dimensions
    canvas_width = cols * card_width + (cols + 1) * margin
    canvas_height = rows * card_height + (rows + 1) * margin
    
    # Calculate scaling factors for Cairo drawing
    original_card_width = 120
    original_card_height = 80
    scale_x = card_width / original_card_width
    scale_y = card_height / original_card_height
    scale_factor = min(scale_x, scale_y)
    
    # Create Cairo surface and context
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_width, canvas_height)
    ctx = cairo.Context(surface)
    
    # Fill background
    ctx.set_source_rgb(0.92, 0.92, 0.94)  # Light gray background like env.py
    ctx.paint()
    
    # Draw each card
    for i, code in enumerate(codes):
        row = i // cols
        col = i % cols
        
        card_x = margin + col * (card_width + margin)
        card_y = margin + row * (card_height + margin)
        
        shape, color, shading, number = code_to_cairo_params(code)
        card_number = i + 1  # 1-based numbering
        _draw_card(ctx, card_x, card_y, shape, color, shading, number, 
                  card_width, card_height, scale_factor, card_number)
    
    # Convert Cairo surface to PIL Image
    buf = surface.get_data()
    img = Image.frombuffer("RGBA", (canvas_width, canvas_height), buf, "raw", "BGRA", 0, 1)
    img = img.convert("RGB")  # Convert to RGB to match env.py
    
    # Add index overlays if requested
    if overlay_indices:
        for i, code in enumerate(codes):
            row = i // cols
            col = i % cols
            
            card_x = margin + col * (card_width + margin)
            card_y = margin + row * (card_height + margin)
            
            _draw_index_overlay(ctx, card_x, card_y, i, change_to_1_based=True)
    
    return img

def _draw_index_overlay(ctx, x, y, index, change_to_1_based=False):
    """Draw index overlay on card using Cairo."""
    # Use 1-based indexing if requested
    display_index = index + 1 if change_to_1_based else index
    
    # Index pill styling
    text = str(display_index)
    
    # Set larger font size for better visibility
    font_size = 24  # Increased from default
    ctx.set_font_size(font_size)
    
    # Get text extents
    text_extents = ctx.text_extents(text)
    text_width = text_extents.width
    text_height = text_extents.height
    
    # Pill dimensions with padding
    pad = 8
    pill_width = text_width + 2 * pad
    pill_height = font_size + pad
    
    # Draw rounded rectangle background
    pill_x = x + 8
    pill_y = y + 8
    
    # Create rounded rectangle path
    radius = 12
    ctx.new_path()
    ctx.arc(pill_x + radius, pill_y + radius, radius, math.pi, 3 * math.pi / 2)
    ctx.arc(pill_x + pill_width - radius, pill_y + radius, radius, 3 * math.pi / 2, 0)
    ctx.arc(pill_x + pill_width - radius, pill_y + pill_height - radius, radius, 0, math.pi / 2)
    ctx.arc(pill_x + radius, pill_y + pill_height - radius, radius, math.pi / 2, math.pi)
    ctx.close_path()
    
    # Fill with semi-transparent black
    ctx.set_source_rgba(0, 0, 0, 0.7)
    ctx.fill_preserve()
    
    # Draw text
    ctx.set_source_rgba(1, 1, 1, 1)  # White text
    ctx.move_to(pill_x + pad, pill_y + font_size - 4)
    ctx.show_text(text)

def generate_all_cards():
    """Generate all 81 unique SET card combinations (for compatibility)."""
    shapes = ['diamond', 'squiggle', 'oval']
    colors = ['green', 'blue', 'red']
    shadings = ['empty', 'striped', 'solid']
    numbers = [1, 2, 3]
    
    return list(itertools.product(shapes, colors, shadings, numbers))

if __name__ == "__main__":
    # Generate all 81 cards using the old method for compatibility
    all_cards = generate_all_cards()
    
    # Create all card codes (0-80)
    all_codes = list(range(81))
    
    # Generate board image using new utility function (9x9 grid for all 81 cards)
    board_img = generate_board_image(all_codes, card_width=480, card_height=320, 
                                   margin=5, cols=27, rows=3)
    
    # Save as PNG
    board_img.save('set_board_81_cards.png')
    print("Generated set_board_81_cards.png with 81 unique SET cards using Cairo")