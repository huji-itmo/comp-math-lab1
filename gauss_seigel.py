from typing import List, Tuple, Optional


def read_augmented_matrix() -> Tuple[List[List[float]], List[float]]:
    """Read augmented matrix [A|b] from user input."""
    n = int(input("Enter matrix size (N): "))
    print(f"\nEnter {n} rows, each with {n+1} numbers (coefficients + constant):")
    A = []
    b = []
    for i in range(n):
        while True:
            row = input(f"Row {i+1}: ").split()
            if len(row) != n + 1:
                print(f"Error: Expected {n+1} elements, got {len(row)}")
                continue
            try:
                nums = [float(x) for x in row]
                A.append(nums[:-1])
                b.append(nums[-1])
                break
            except ValueError:
                print("Invalid input! Please enter numbers only.")
    return A, b


def read_matrix_from_file(
    filename: str,
) -> Optional[Tuple[List[List[float]], List[float]]]:
    """Read matrix from file with format: first line N, then N rows of N+1 numbers."""
    print("\nFile format requirements:")
    print("- First line: matrix size N (positive integer)")
    print("- Next N lines: rows with N+1 numbers separated by spaces")
    print("Example file content:")
    print("3")
    print("1.0 2.0 3.0 2.0")
    print("4.5 5.0 6.7 2.0")
    print("7.0 8.8 9.9 2.0")

    try:
        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                print("Error: Empty file")
                return None

            n = int(lines[0])
            if len(lines) != n + 1:
                print(f"Error: Expected {n+1} lines, got {len(lines)}")
                return None

            A = []
            b = []
            for line in lines[1:]:
                nums = [float(x) for x in line.split()]
                if len(nums) != n + 1:
                    print(f"Error: Row contains {len(nums)} elements, expected {n+1}")
                    return None
                A.append(nums[:-1])
                b.append(nums[-1])
            return A, b
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None


def is_diagonally_dominant(A: List[List[float]]) -> bool:
    """Check if matrix is strictly diagonally dominant."""
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= row_sum:
            return False
    return True


def rearrange_matrix(
    A: List[List[float]], b: List[float]
) -> Tuple[bool, List[List[float]], List[float], List[int]]:
    """Attempt to make matrix diagonally dominant through row/column swaps."""
    n = len(A)
    col_order = list(range(n))

    for i in range(n):
        # Row swaps
        max_row = i
        max_val = abs(A[i][i])
        for j in range(i, n):
            if abs(A[j][i]) > max_val:
                max_val = abs(A[j][i])
                max_row = j
        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]

        # Check dominance
        diag = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag > row_sum:
            continue

        # Column swaps
        max_col = i
        max_val = abs(A[i][i])
        for j in range(i, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                max_col = j
        if max_col != i:
            for row in A:
                row[i], row[max_col] = row[max_col], row[i]
            col_order[i], col_order[max_col] = col_order[max_col], col_order[i]

        # Final check
        diag = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= row_sum:
            return False, A, b, col_order

    return is_diagonally_dominant(A), A, b, col_order


def matrix_inf_norm(A: List[List[float]]) -> float:
    """Calculate infinity norm of matrix (maximum row sum)."""
    return max(sum(abs(x) for x in row) for row in A)


def gauss_seidel(
    A: List[List[float]], b: List[float], epsilon: float, max_iter: int = 1000
) -> Tuple[List[float], int, List[float]]:
    """Perform Gauss-Seidel iteration."""
    n = len(A)
    x = [0.0] * n
    iterations = 0
    errors = []

    for _ in range(max_iter):
        x_prev = x.copy()
        max_error = 0.0

        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]

        current_errors = [abs(x[i] - x_prev[i]) for i in range(n)]
        errors.append(current_errors)
        iterations += 1

        if max(current_errors) < epsilon:
            break

    return x, iterations, errors[-1]


def main():
    print("Gauss-Seidel Method Solver")
    print("===========================")

    # Input selection
    choice = input("Choose input method (1-manual, 2-file): ")
    if choice == "1":
        A, b = read_augmented_matrix()
    elif choice == "2":
        filename = input("Enter filename: ")
        result = read_matrix_from_file(filename)
        if not result:
            return
        A, b = result
    else:
        print("Invalid choice")
        return

    n = len(A)

    # Check and rearrange matrix
    success, A, b, col_order = rearrange_matrix(A, b)
    if not success:
        print("Warning: Could not achieve diagonal dominance!")

    # Get precision
    while True:
        try:
            epsilon = float(input("\nEnter desired precision (e.g. 0.001): "))
            if epsilon <= 0:
                print("Please enter positive value")
                continue
            break
        except ValueError:
            print("Invalid input! Enter a number.")

    # Solve system
    x, iterations, errors = gauss_seidel(A, b, epsilon)

    # Reorder solution if columns were swapped
    x_original = [0.0] * n
    for original_col in range(n):
        for new_col in range(n):
            if col_order[new_col] == original_col:
                x_original[original_col] = x[new_col]
                break

    # Output results
    print("\nResults:")
    print(f"Matrix infinity norm: {matrix_inf_norm(A):.4f}")
    print(f"Iterations required: {iterations}")

    print("\nSolution vector:")
    for i, val in enumerate(x_original, 1):
        print(f"x{i} = {val:.6f}")

    print("\nFinal iteration errors:")
    for i, err in enumerate(errors, 1):
        print(f"Δx{i} = {err:.6f}")


if __name__ == "__main__":
    main()
