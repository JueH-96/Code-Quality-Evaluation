import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    equations = []
    for ak in A:
        # Compute the parity of A_k (number of 1's mod 2)
        parity_ak = bin(ak).count('1') % 2
        # Coefficients are 1 for bits where A_k is 0
        coeff = [0] * 100
        for i in range(100):
            if not (ak & (1 << i)):
                coeff[i] = 1
        # The equation is sum(coeff & x) ≡ (1 + parity_ak) mod 2
        constant = (1 + parity_ak) % 2
        equations.append((coeff, constant))
    
    # Build the matrix for Gaussian elimination
    matrix = []
    for coeff, c in equations:
        row = coeff.copy()
        row.append(c)  # Add the constant term at the end
        matrix.append(row)
    
    n_vars = 100
    rank = 0
    pivot_cols = []
    
    # Perform Gaussian elimination over GF(2)
    for col in range(n_vars):
        # Find the pivot row
        pivot_row = -1
        for r in range(rank, len(matrix)):
            if matrix[r][col]:
                pivot_row = r
                break
        if pivot_row == -1:
            continue  # No pivot in this column, move to next
        # Swap current row with the pivot row
        matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
        # Eliminate all other rows
        for r in range(len(matrix)):
            if r != rank and matrix[r][col]:
                for c2 in range(col, n_vars + 1):
                    matrix[r][c2] ^= matrix[rank][c2]
        pivot_cols.append(col)
        rank += 1
    
    # Count the number of inconsistent equations (rows of the form 0=1)
    inconsistent = 0
    for row in matrix[rank:]:
        if row[-1]:
            inconsistent += 1
    
    # The maximum number of satisfied equations is total equations minus inconsistent ones
    max_satisfied = len(equations) - inconsistent
    print(max_satisfied)

if __name__ == '__main__':
    main()