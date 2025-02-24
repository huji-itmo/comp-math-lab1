from typing import List, Optional, Tuple

from system_of_linear_equations import SystemOfLinearEquations


def get_augmented_matrix(n: int) -> SystemOfLinearEquations:
    """Get and validate the augmented matrix (coefficients + constants) from the user."""
    augmented_matrix: List[List[float]] = []
    print(
        f"\nEnter {n} rows, each containing {n+1} numbers (coefficients followed by constant):"
    )
    for i in range(n):
        while True:
            row_input: str = input(f"Row {i+1}: ").strip()
            elements: List[str] = row_input.split()
            if len(elements) != n + 1:
                print(
                    f"Error: Expected {n+1} elements, got {len(elements)}. Please try again."
                )
                continue
            try:
                row: List[float] = [float(x) for x in elements]
                augmented_matrix.append(row)
                break
            except ValueError:
                print("Error: Please enter valid numbers only. Try again.")

    return SystemOfLinearEquations(augmented_matrix)


def read_augmented_matrix_from_file(
    filename: str,
) -> Optional[SystemOfLinearEquations]:
    """Read augmented matrix from a properly formatted file and split into matrix and constants."""
    try:
        with open(filename, "r") as f:
            lines: List[str] = [line.strip() for line in f if line.strip()]

            if not lines:
                print("Error: Empty file")
                return None

            try:
                n: int = int(lines[0])
            except ValueError:
                print(
                    "Error: First line must be a single integer (number of variables)"
                )
                return None

            if len(lines) != n + 1:
                print(f"Error: Expected {n+1} lines, got {len(lines)}")
                return None

            augmented_matrix: List[List[float]] = []
            for line in lines[1:]:
                elements: List[str] = line.split()
                if len(elements) != n + 1:
                    print(
                        f"Error: Row contains {len(elements)} elements, expected {n+1}"
                    )
                    return None
                try:
                    row: List[float] = [float(x) for x in elements]
                    augmented_matrix.append(row)
                except ValueError:
                    print(f"Error: Invalid number format in row: {line}")
                    return None

            return SystemOfLinearEquations(augmented_matrix)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None


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


def get_matrix() -> SystemOfLinearEquations:
    print("Matrix Input Program")
    print("====================")

    augmented_matrix: SystemOfLinearEquations

    while True:

        choice: str = input(
            "\nChoose input method:\n"
            "1. Manual entry\n"
            "2. File input\n"
            "Enter choice (1/2): "
        ).strip()

        if choice == "1":
            n: int = get_matrix_size()
            augmented_matrix: SystemOfLinearEquations = get_augmented_matrix(n)
            break
        elif choice == "2":
            print("\nFile format requirements:")
            print("- First line: matrix size N (positive integer)")
            print("- Next N lines: rows with N numbers separated by spaces")
            print("Example file content:")
            print("3")
            print("1.0 2.0 3.0 2.0")
            print("4.5 5.0 6.7 2.0")
            print("7.0 8.8 9.9 2.0")

            while True:
                filename: str = input("\nEnter filename: ").strip()
                augmented_matrix: SystemOfLinearEquations | None = (
                    read_augmented_matrix_from_file(filename)
                )
                if augmented_matrix is None:
                    print("Please try another file or check the format.")
                else:
                    break

            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    return augmented_matrix
