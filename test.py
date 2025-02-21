from typing import Optional
import pytest  # type: ignore
from gauss_seigel import gauss_seidel, rearrange_matrix


def get_res(A, b, max_iter=50, epsilon=1e-6) -> Optional[list[float]]:
    success, A, b, col_order = rearrange_matrix(A, b)
    x, iterations, errors = gauss_seidel(A, b, epsilon, max_iter)
    n = len(A)
    # Reorder solution if columns were swapped
    x_original = [0.0] * n
    for original_col in range(n):
        for new_col in range(n):
            if col_order[new_col] == original_col:
                x_original[original_col] = x[new_col]
                break

    if iterations == max_iter:
        return None

    return x_original


def test_3x3_diagonally_dominant():
    A = [[4.0, 1.0, 1.0], [1.0, 5.0, 2.0], [0.0, 1.0, 3.0]]
    b = [6.0, 8.0, 4.0]
    expected = [1.0, 1.0, 1.0]
    result = get_res(A, b, max_iter=50, epsilon=1e-1)
    assert result == pytest.approx(expected, abs=1e-2)


def test_3x3_symmetric_positive_definite():
    A = [[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]]
    b = [5.0, 6.0, 5.0]
    expected = [1.0, 1.0, 1.0]
    result = get_res(A, b, max_iter=50, epsilon=1e-6)
    assert result == pytest.approx(expected, abs=1e-6)


def test_3x3_non_diagonally_dominant():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    b = [2.0, 5.0, 8.0]
    expected = [1.0, -1.0, 1.0]

    # This may not converge - test with lower precision and fewer iterations
    result = get_res(A, b, max_iter=20, epsilon=1e-3)
    # assert result == pytest.approx(expected, abs=1e-1)
    assert result == None


def test_4x4_sparse_diagonally_dominant():
    A = [
        [5.0, 0.0, 0.0, 1.0],
        [0.0, 6.0, 2.0, 0.0],
        [0.0, 1.0, 7.0, 0.0],
        [1.0, 0.0, 0.0, 8.0],
    ]
    b = [6.0, 8.0, 8.0, 9.0]
    expected = [1.0, 1.0, 1.0, 1.0]
    result = get_res(A, b, max_iter=100, epsilon=1e-6)
    assert result == pytest.approx(expected, abs=1e-6)


def test_2x2_symmetric_diagonally_dominant():
    A = [[3.0, 1.0], [1.0, 3.0]]
    b = [4.0, 4.0]
    expected = [1.0, 1.0]
    result = get_res(A, b, max_iter=20, epsilon=1e-6)
    assert result == pytest.approx(expected, abs=1e-6)


def test_4x4_large_diagonally_dominant():
    A = [
        [10.0, 2.0, 3.0, 1.0],
        [1.0, 12.0, 1.0, 2.0],
        [2.0, 1.0, 15.0, 3.0],
        [1.0, 2.0, 1.0, 20.0],
    ]
    b = [16.0, 16.0, 21.0, 24.0]
    expected = [1.0, 1.0, 1.0, 1.0]
    result = get_res(A, b, max_iter=100, epsilon=1e-6)
    assert result == pytest.approx(expected, abs=1e-6)


def test_3x3_hilbert_slow_convergence():
    A = [[1.0, 0.5, 0.3333], [0.5, 0.3333, 0.25], [0.3333, 0.25, 0.2]]
    b = [1.8333, 1.0833, 0.7833]
    expected = [1.0, 1.0, 1.0]

    # Allow more iterations and lower precision
    result = get_res(A, b, max_iter=500, epsilon=1e-4)
    assert result == pytest.approx(expected, abs=1e-3)


def test_3x3_diagonal_trivial():
    A = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    b = [5.0, 5.0, 5.0]
    expected = [1.0, 1.0, 1.0]
    result = get_res(A, b, max_iter=1, epsilon=1e-6)
    assert result == pytest.approx(expected, abs=1e-6)


if __name__ == "__main__":
    test_3x3_diagonally_dominant()
    test_3x3_symmetric_positive_definite()
    test_3x3_non_diagonally_dominant()
    test_4x4_sparse_diagonally_dominant()
    test_2x2_symmetric_diagonally_dominant()
    test_4x4_large_diagonally_dominant()
    test_3x3_hilbert_slow_convergence()
    test_3x3_diagonal_trivial()
