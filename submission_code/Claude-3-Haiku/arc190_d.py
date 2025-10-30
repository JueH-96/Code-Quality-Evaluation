import sys
from itertools import product

def solve():
    N, p = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(N)]

    # Find the positions of zeros in A
    zero_positions = [(i, j) for i in range(N) for j in range(N) if A[i][j] == 0]
    num_zeros = len(zero_positions)

    # Compute the sum of B^p for all possible B
    result = [[0] * N for _ in range(N)]
    for b_values in product(range(1, p), repeat=num_zeros):
        b = A.copy()
        for (i, j), val in zip(zero_positions, b_values):
            b[i][j] = val
        b_power_p = [[pow(b[i][j], p, p) for j in range(N)] for i in range(N)]
        for i in range(N):
            for j in range(N):
                result[i][j] = (result[i][j] + b_power_p[i][j]) % p

    # Print the result
    for row in result:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    solve()