from typing import List
from matrix_reader import get_matrix
from system_of_linear_equations import SystemOfLinearEquations


if __name__ == "__main__":
    matrix: SystemOfLinearEquations = get_matrix()

    print(matrix)

    while True:
        try:
            epsilon: float = float(input("\nEnter precision (small number epsilon): "))
            if epsilon < 0:
                print("Please enter a positive float!")

            break
        except ValueError:
            print("Invalid input! Please enter a valid float.")

    success_rearranging: bool = matrix.rearrange_matrix()

    if success_rearranging:
        print("Successfully rearranged matrix to make it diagonally dominant.")
    else:
        print("Failed to rearrange matrix to make it diagonally dominant.")

    x, iterations, errors = matrix.gauss_seidel(epsilon, 1000)

    # Output results
    print("\nResults:")
    print(f"Matrix infinity norm: {matrix.matrix_inf_norm():.4f}")
    print(f"Iterations required: {iterations}")

    print("\nSolution vector:")
    for i, val in enumerate(x, 1):
        print(f"x{i} = {val:.6f}")

    print("\nFinal iteration errors:")
    for i, err in enumerate(errors, 1):
        print(f"Δx{i} = {err:.6f}")
