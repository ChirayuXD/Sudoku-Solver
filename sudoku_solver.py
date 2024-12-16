import cv2
import numpy as np
import pytesseract
import os
from typing import List, Tuple, Optional

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Directory to save output images
output_dir = "sudoku_output_images"
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    
    max_size = 1000
    height, width = img.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite(os.path.join(output_dir, "1_preprocessed_image.jpg"), thresh)
    return thresh, img

def find_corners(contour: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect)
        corners = np.int0(corners)
        return corners
    
    return approx.reshape(-1, 2)

def order_corners(corners: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    d = np.diff(corners, axis=1)
    
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    rect[1] = corners[np.argmin(d)]
    rect[3] = corners[np.argmax(d)]
    
    return rect

def detect_grid(thresh_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(
        thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    grid_contour = max(contours, key=cv2.contourArea)
    corners = find_corners(grid_contour)
    corners = order_corners(corners)
    
    grid_size = 450
    dst_points = np.array([
        [0, 0],
        [grid_size - 1, 0],
        [grid_size - 1, grid_size - 1],
        [0, grid_size - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    warped = cv2.warpPerspective(thresh_image, matrix, (grid_size, grid_size))
    
    cv2.imwrite(os.path.join(output_dir, "2_warped_grid.jpg"), warped)
    return warped, matrix

def extract_digits(warped_grid: np.ndarray) -> List[List[int]]:
    cell_size = warped_grid.shape[0] // 9
    board = [[0 for _ in range(9)] for _ in range(9)]
    
    for i in range(9):
        for j in range(9):
            cell = warped_grid[
                i * cell_size:(i + 1) * cell_size,
                j * cell_size:(j + 1) * cell_size
            ]
            
            padding = cell_size // 8
            cell = cell[padding:-padding, padding:-padding]
            
            if np.sum(cell) < cell.size * 255 * 0.1:
                continue
            
            config = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"
            digit = pytesseract.image_to_string(cell, config=config).strip()
            
            if digit and digit.isdigit() and 1 <= int(digit) <= 9:
                board[i][j] = int(digit)
            
            # Save each cell as an image for review
            cell_path = os.path.join(output_dir, f"3_cell_{i}_{j}.jpg")
            cv2.imwrite(cell_path, cell)
    
    return board

def display_solution(original_image: np.ndarray, warped: np.ndarray, 
                    solved_board: List[List[int]], transform_matrix: np.ndarray) -> None:
    height, width = warped.shape
    solution_img = np.zeros((height, width, 3), dtype=np.uint8)
    cell_size = height // 9
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell_size / 50.0
    thickness = max(1, int(cell_size / 25.0))
    
    for i in range(9):
        for j in range(9):
            if solved_board[i][j] != 0:
                text = str(solved_board[i][j])
                textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = j * cell_size + (cell_size - textsize[0]) // 2
                text_y = i * cell_size + (cell_size + textsize[1]) // 2
                cv2.putText(solution_img, text, (text_x, text_y),
                           font, font_scale, (0, 255, 0), thickness)
    
    inv_matrix = cv2.invert(transform_matrix)[1]
    unwarped_solution = cv2.warpPerspective(solution_img, inv_matrix, 
                                          (original_image.shape[1], original_image.shape[0]))
    
    mask = (unwarped_solution != 0).any(axis=2)
    result = original_image.copy()
    result[mask] = cv2.addWeighted(original_image[mask], 0.4, unwarped_solution[mask], 0.6, 0)
    
    cv2.imwrite(os.path.join(output_dir, "4_final_solution_overlay.jpg"), result)
def is_valid_board(board: List[List[int]]) -> bool:
    """
    Check if the recognized board is valid.
    
    Args:
        board: 9x9 Sudoku board
    Returns:
        True if board is valid, False otherwise
    """
    def is_valid_unit(unit):
        unit = [x for x in unit if x != 0]
        return len(unit) == len(set(unit))
    
    # Check rows
    for row in board:
        if not is_valid_unit(row):
            return False
    
    # Check columns
    for col in zip(*board):
        if not is_valid_unit(col):
            return False
    
    # Check 3x3 boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = []
            for k in range(3):
                for l in range(3):
                    box.append(board[i + k][j + l])
            if not is_valid_unit(box):
                return False
    
    return True

def solve_sudoku(board: List[List[int]]) -> bool:
    """
    Solve the Sudoku puzzle using backtracking.
    
    Args:
        board: 9x9 Sudoku board
    Returns:
        True if solution found, False otherwise
    """
    empty = find_empty_location(board)
    if not empty:
        return True
    
    row, col = empty
    for num in range(1, 10):
        if is_safe(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    
    return False

def find_empty_location(board: List[List[int]]) -> Optional[Tuple[int, int]]:
    """Find an empty location in the board."""
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_safe(board: List[List[int]], row: int, col: int, num: int) -> bool:
    """Check if it's safe to place a number in a cell."""
    # Check row
    if num in board[row]:
        return False
    
    # Check column
    if num in [board[i][col] for i in range(9)]:
        return False
    
    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if board[i][j] == num:
                return False
    
    return True

def display_solution(original_image: np.ndarray, warped: np.ndarray, 
                    solved_board: List[List[int]], transform_matrix: np.ndarray) -> None:
    """
    Display the solution overlaid on the original image.
    
    Args:
        original_image: Original input image
        warped: Warped grid image
        solved_board: Solved Sudoku board
        transform_matrix: Perspective transform matrix
    """
    # Create blank image for solution
    height, width = warped.shape
    solution_img = np.zeros((height, width, 3), dtype=np.uint8)
    cell_size = height // 9
    
    # Add solved digits
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell_size / 50.0
    thickness = max(1, int(cell_size / 25.0))
    
    for i in range(9):
        for j in range(9):
            if solved_board[i][j] != 0:
                text = str(solved_board[i][j])
                textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = j * cell_size + (cell_size - textsize[0]) // 2
                text_y = i * cell_size + (cell_size + textsize[1]) // 2
                cv2.putText(solution_img, text, (text_x, text_y),
                           font, font_scale, (0, 255, 0), thickness)
    
    # Inverse perspective transform
    inv_matrix = cv2.invert(transform_matrix)[1]
    unwarped_solution = cv2.warpPerspective(solution_img, inv_matrix, 
                                          (original_image.shape[1], original_image.shape[0]))
    
    # Combine with original image
    mask = (unwarped_solution != 0).any(axis=2)
    result = original_image.copy()
    result[mask] = cv2.addWeighted(original_image[mask], 0.4, unwarped_solution[mask], 0.6, 0)
    
    # Display result
    cv2.imshow("Original", original_image)
    cv2.imshow("Solved Sudoku", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import time

def main(image_path: str) -> None:
    """
    Main function to run the Sudoku solver.
    
    Args:
        image_path: Path to the input image
    """
    try:
        # Preprocess image
        thresh, original_image = preprocess_image(image_path)
        
        # Detect and extract grid
        warped_grid, transform_matrix = detect_grid(thresh)
        
        # Extract digits
        board = extract_digits(warped_grid)
        
        # Validate board
        if not is_valid_board(board):
            raise ValueError("Invalid Sudoku board detected")
        
        # Measure time for solving using backtracking
        start_time = time.time()
        solved = solve_sudoku(board)
        end_time = time.time()
        
        # Calculate backtracking runtime
        backtracking_time = end_time - start_time
        
        # Check and display results
        if solved:
            display_solution(original_image, warped_grid, board, transform_matrix)
            print(f"Backtracking algorithm runtime: {backtracking_time:.4f} seconds")
        else:
            print("No solution exists for the detected puzzle")
        
    except Exception as e:
        print(f"Error processing Sudoku puzzle: {str(e)}")

if __name__ == "__main__":
    image_path = "C:\\Users\\chira\\Downloads\\sudoku_hard.jpg"  # Change this to your image path
    main(image_path)

