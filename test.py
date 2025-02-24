from typing import Optional
import pytest

from system_of_linear_equations import SystemOfLinearEquations  # type: ignore


def get_res(
    matrix: SystemOfLinearEquations, max_iter=50, epsilon=1e-6
) -> Optional[list[float]]:
    print(f"-" * 32)
    success = matrix.rearrange_matrix()
    x, iterations, errors = matrix.gauss_seidel(epsilon, max_iter)
    if not success:
        print(f"Failed to make matrix diagonally dominant.")
        # return None
    print(matrix)
    print(f"Result: {x}")
    print(f"Iterations: {iterations}")

    if iterations == max_iter:
        print(f"Reached iteration limit: {max_iter}")
        return None

    return x


def test_3x3_diagonally_dominant():
    A = [[4.0, 1.0, 1.0], [1.0, 5.0, 2.0], [0.0, 1.0, 3.0]]
    b = [6.0, 8.0, 4.0]
    expected = [1.0, 1.0, 1.0]
    result = get_res(SystemOfLinearEquations(A, b), max_iter=500, epsilon=1e-2)
    assert result == pytest.approx(expected, abs=1e-2)


def test_3x3_symmetric_positive_definite():
    A = [[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]]
    b = [5.0, 6.0, 5.0]
    expected = [1.0, 1.0, 1.0]
    result = get_res(SystemOfLinearEquations(A, b), max_iter=500, epsilon=1e-6)
    assert result == pytest.approx(expected, abs=1e-6)


def test_3x3_non_diagonally_dominant():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    b = [2.0, 5.0, 8.0]

    expected = [1.0, -1.0, 1.0]

    # doesn't converge
    result = get_res(SystemOfLinearEquations(A, b), max_iter=500, epsilon=1e-3)
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
    result = get_res(SystemOfLinearEquations(A, b), max_iter=100, epsilon=1e-6)
    assert result == pytest.approx(expected, abs=1e-6)


def test_2x2_symmetric_diagonally_dominant():
    A = [[3.0, 1.0], [1.0, 3.0]]
    b = [4.0, 4.0]
    expected = [1.0, 1.0]
    result = get_res(SystemOfLinearEquations(A, b), max_iter=500, epsilon=1e-6)
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
    result = get_res(SystemOfLinearEquations(A, b), max_iter=100, epsilon=1e-6)
    assert result == pytest.approx(expected, abs=1e-6)


def test_3x3_hilbert_slow_convergence():
    A = [[1.0, 0.5, 0.3333], [0.5, 0.3333, 0.25], [0.3333, 0.25, 0.2]]
    b = [1.8333, 1.0833, 0.7833]
    expected = [1.0, 1.0, 1.0]

    # Allow more iterations and lower precision
    result = get_res(SystemOfLinearEquations(A, b), max_iter=5000, epsilon=1e-6)
    # if result == None:
    assert result == pytest.approx(expected, abs=1e-4)


def test_3x3_diagonal_trivial():
    A = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    b = [5.0, 5.0, 5.0]
    expected = [1.0, 1.0, 1.0]
    result = get_res(SystemOfLinearEquations(A, b), max_iter=100, epsilon=1e-6)
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
