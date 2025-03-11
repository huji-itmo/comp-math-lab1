from typing import Final, List, Tuple


class SystemOfLinearEquations:
    matrix: List[List[float]]
    constants: List[float]

    side_length: int

    new_column_order: List[int]

    debug: bool = True

    def matrix_inf_norm(self) -> float:
        """Calculate infinity norm of matrix (maximum row sum)."""
        return max(sum(abs(x) for x in row) for row in self.matrix)

    def matrix_special_norm(self) -> float:
        """Calculate infinity norm of matrix (maximum row sum)."""
        self.rearrange_matrix()
        c_matrix = []
        for i in range(self.side_length):
            row = [0.0] * self.side_length
            for j in range(self.side_length):
                if i != j:
                    row[j] = self.matrix[j][i] / self.matrix[i][i]
            c_matrix.append(row)

        return max(sum(abs(x) for x in row) for row in c_matrix)

    def __init__(
        self,
        augmented_or_square_matrix: List[List[float]],
        constants: List[float] | None = None,
    ):
        if constants is None:
            constants = [row[-1] for row in augmented_or_square_matrix]
            augmented_or_square_matrix = [
                row[:-1] for row in augmented_or_square_matrix
            ]

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

        if self.debug:
            print("-" * 32)
            print(f"Swap rows {row1} and {row2}")
            print(self)

    def swap_columns(self, col1: int, col2: int) -> None:
        """Swap two columns in the matrix."""
        for row in self.matrix:
            row[col1], row[col2] = row[col2], row[col1]

        self.new_column_order[col1], self.new_column_order[col2] = (
            self.new_column_order[col2],
            self.new_column_order[col1],
        )

        if self.debug:
            print("-" * 32)
            print(f"Swap coumns {col1} and {col2}")
            print(self)

    def is_diagonally_dominant(self) -> bool:
        """Check if matrix is strictly diagonally dominant."""
        n: int = self.side_length
        for i in range(n):
            diag = abs(self.matrix[i][i])
            row_sum = sum(abs(self.matrix[i][j]) for j in range(n) if j != i)
            if diag < row_sum:
                return False
        return True

    def rearrange_matrix(self) -> bool:
        """Attempt to make matrix diagonally dominant through row/column swaps."""
        n: int = self.side_length

        if self.debug:
            print("Initial state of matrix")
            print(self)

        for i in range(n):
            # Row swaps
            max_row = i
            max_val = abs(self.matrix[i][i])
            for j in range(i, n):
                current: float = abs(self.matrix[j][i])
                if current > max_val:
                    max_val = current
                    max_row = j
            if max_row != i:
                self.swap_rows(max_row, i)

            # Check dominance
            diag = abs(self.matrix[i][i])
            row_sum = sum(abs(self.matrix[i][j]) for j in range(n) if j != i)
            if diag >= row_sum:
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
            if diag < row_sum:
                return False

        return self.is_diagonally_dominant()

    def __str__(self) -> str:
        """
        Returns a string representation of the system of linear equations in a matrix-like format,
        with equal spacing between columns.
        """

        # Determine the format string for each float
        float_format = f"{{:>{8}.{2}f}}"
        formatted_rows = list[str]()
        for row in self.matrix:
            # Format each element in the row and join them with spaces
            formatted_rows.append(" ".join(float_format.format(val) for val in row))

        return "\n".join(formatted_rows) + "\n" + str(self.constants)

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

    def compute_residuals(self, answer_vector: List[float]) -> List[float]:
        residuals = []
        for i in range(self.side_length):
            # Вычисляем невязку для i-го уравнения
            residual = -self.constants[i]
            for j in range(self.side_length):
                residual += self.matrix[i][j] * answer_vector[j]
            residuals.append(residual)
        return residuals


def rearrange_according_to_order_list(
    input: List[float], order: List[int]
) -> List[float]:
    n: int = len(order)
    res = []

    for i in range(n):
        res.append(input[order[i]])

    return res
