from typing import List, Optional

def get_matrix_size() -> int:
    """Get and validate the matrix size from the user."""
    while True:
        try:
            n: int = int(input("\nEnter the side length of the matrix (N): "))
            if n > 0:
                return n
            print("Please enter a positive integer!")
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

def get_matrix(n: int) -> List[List[float]]:
    """Get and validate matrix elements from the user."""
    matrix: List[List[float]] = []
    print(f"\nEnter {n} rows, each containing {n} numbers separated by spaces:")

    for i in range(n):
        while True:
            row_input: str = input(f"Row {i+1}: ").strip()
            elements: List[str] = row_input.split()

            if len(elements) != n:
                print(f"Error: Expected {n} elements, got {len(elements)}. Please try again.")
                continue

            try:
                row: List[float] = [float(x) for x in elements]
                matrix.append(row)
                break
            except ValueError:
                print("Error: Please enter valid numbers only. Try again.")

    return matrix

def read_matrix_from_file(filename: str) -> Optional[List[List[float]]]:
    """Read matrix from a properly formatted file."""
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

            if not lines:
                print("Error: Empty file")
                return None

            try:
                n = int(lines[0])
            except ValueError:
                print("Error: First line must be a single integer (matrix size)")
                return None

            if len(lines) != n + 1:
                print(f"Error: Expected {n+1} lines, got {len(lines)}")
                return None

            matrix: List[List[float]] = []
            for line in lines[1:]:
                elements = line.split()
                if len(elements) != n:
                    print(f"Error: Row contains {len(elements)} elements, expected {n}")
                    return None
                try:
                    matrix.append([float(x) for x in elements])
                except ValueError:
                    print(f"Error: Invalid number format in row: {line}")
                    return None

            return matrix

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None

def print_matrix(matrix: List[List[float]]) -> None:
    """Display the matrix in a clean, formatted layout."""
    print("\nMatrix successfully entered:")
    for row in matrix:
        formatted_row: List[str] = [f"{num:8.2f}" for num in row]
        print("  ".join(formatted_row))

def main() -> None:
    print("Matrix Input Program")
    print("====================")

    while True:
        choice: str = input(
            "\nChoose input method:\n"
            "1. Manual entry\n"
            "2. File input\n"
            "Enter choice (1/2): "
        ).strip()

        if choice == '1':
            n: int = get_matrix_size()
            matrix: List[List[float]] = get_matrix(n)
            break
        elif choice == '2':
            print("\nFile format requirements:")
            print("- First line: matrix size N (positive integer)")
            print("- Next N lines: rows with N numbers separated by spaces")
            print("Example file content:")
            print("3")
            print("1.0 2.0 3.0")
            print("4.5 5.0 6.7")
            print("7.0 8.8 9.9")

            while True:
                filename: str = input("\nEnter filename: ").strip()
                matrix = read_matrix_from_file(filename)
                if matrix is not None:
                    n = len(matrix)
                    break
                print("Please try another file or check the format.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    print_matrix(matrix)

if __name__ == "__main__":
    main()
