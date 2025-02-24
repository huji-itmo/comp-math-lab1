from typing import Final, List, Tuple


class SystemOfLinearEquations:
    matrix: List[List[float]]
    constants: List[float]

    side_length: int

    new_column_order: List[int]

    def matrix_inf_norm(self) -> float:
        """Calculate infinity norm of matrix (maximum row sum)."""
        return max(sum(abs(x) for x in row) for row in self.matrix)

    def __init__(
        self,
        augmented_or_square_matrix: List[List[float]],
        constants: List[float] | None = None,
    ):
        if constants is None:
            augmented_or_square_matrix = [
                row[:-1] for row in augmented_or_square_matrix
            ]
            constants = [row[-1] for row in augmented_or_square_matrix]

        self.side_length = len(augmented_or_square_matrix)
        self.matrix = augmented_or_square_matrix
        self.constants = constants

        self.new_column_order = list(range(self.side_length))

    def swap_rows(self, row1: int, row2: int) -> None:
        """Swap two rows in the matrix and their corresponding constants."""
        self.matrix[row1], self.matrix[row2] = self.matrix[row2], self.matrix[row1]
        self.constants[row1], self.constants[row2] = (
            self.constants[row2],
            self.constants[row1],
        )

    def swap_columns(self, col1: int, col2: int) -> None:
        """Swap two columns in the matrix."""
        for row in self.matrix:
            row[col1], row[col2] = row[col2], row[col1]

        self.new_column_order[col1], self.new_column_order[col2] = (
            self.new_column_order[col2],
            self.new_column_order[col1],
        )

    def is_diagonally_dominant(self) -> bool:
        """Check if matrix is strictly diagonally dominant."""
        n: int = self.side_length
        for i in range(n):
            diag = abs(self.matrix[i][i])
            row_sum = sum(abs(self.matrix[i][j]) for j in range(n) if j != i)
            if diag <= row_sum:
                return False
        return True

    def rearrange_matrix(self) -> bool:
        """Attempt to make matrix diagonally dominant through row/column swaps."""
        n: int = self.side_length

        for i in range(n):
            # Row swaps
            max_row = i
            max_val = abs(self.matrix[i][i])
            for j in range(i, n):
                current: float = self.matrix[j][i]
                if abs(current) > max_val:
                    max_val = abs(current)
                    max_row = j
            if max_row != i:
                self.swap_rows(max_row, i)

            # Check dominance
            diag = abs(self.matrix[i][i])
            row_sum = sum(abs(self.matrix[i][j]) for j in range(n) if j != i)
            if diag > row_sum:
                continue

            # Column swaps
            max_col = i
            max_val = abs(self.matrix[i][i])
            for j in range(i, n):
                if abs(self.matrix[i][j]) > max_val:
                    max_val = abs(self.matrix[i][j])
                    max_col = j
            if max_col != i:
                self.swap_columns(i, max_col)

            # Final check
            diag = abs(self.matrix[i][i])
            row_sum = sum(abs(self.matrix[i][j]) for j in range(n) if j != i)
            if diag <= row_sum:
                return False

        return self.is_diagonally_dominant()

    def __str__(self) -> str:
        """
        Returns a string representation of the system of linear equations.
        Each equation is displayed in the form: a1*x1 + a2*x2 + ... + an*xn = b
        """
        result = []
        for i in range(self.side_length):
            equation = ""
            for j in range(self.side_length):
                coefficient = self.matrix[i][j]
                # Skip terms with zero coefficients
                if abs(coefficient) == 0:
                    continue
                # Handle positive and negative signs
                sign = "+" if coefficient > 0 else "-"
                coefficient = abs(coefficient)
                # Format the term
                if coefficient == 1:  # Avoid writing '1*' for terms like '+x'
                    term = f" {sign} x{j+1}"
                else:
                    term = f" {sign} {coefficient:.2f}x{j+1}"
                equation += term
            # Add the constant term
            constant = self.constants[i]
            if not equation:  # Handle the case where all coefficients are zero
                equation = f"0 = {constant:.2f}"
            else:
                equation = equation.lstrip(" +") + f" = {constant:.2f}"
            result.append(equation)
        return "\n".join(result)

    def gauss_seidel(
        self, epsilon: float, max_iter: int = 1000
    ) -> Tuple[List[float], int, List[float]]:
        """Perform Gauss-Seidel iteration."""
        n = self.side_length
        answer_vector = [0.0] * n
        iterations = 0
        errors = []

        for _ in range(max_iter):
            x_prev = answer_vector.copy()

            for i in range(n):
                sigma = sum(
                    self.matrix[i][j] * answer_vector[j] for j in range(n) if j != i
                )
                answer_vector[i] = (self.constants[i] - sigma) / self.matrix[i][i]

            current_errors = [abs(answer_vector[i] - x_prev[i]) for i in range(n)]
            errors.append(current_errors)
            iterations += 1

            if max(current_errors) < epsilon:
                break

        answer_vector = rearrange_according_to_order_list(
            answer_vector, self.new_column_order
        )

        return answer_vector, iterations, errors[-1]


def rearrange_according_to_order_list(
    input: List[float], order: List[int]
) -> List[float]:
    n: int = len(order)
    res = []

    for i in range(n):
        res.append(input[order[i]])

    return res
