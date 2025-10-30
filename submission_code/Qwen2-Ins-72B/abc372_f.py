from collections import defaultdict
from functools import lru_cache
from typing import List

MOD = 998244353

def matrix_mod_exp(matrix: List[List[int]], exp: int, mod: int) -> List[List[int]]:
    """
    Computes the power of a matrix modulo a number.
    """
    n = len(matrix)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        result[i][i] = 1

    while exp > 0:
        if exp % 2 == 1:
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        result[i][j] = (result[i][j] + matrix[i][k] * result[k][j]) % mod
        for i in range(n):
            for j in range(n):
                matrix[i][j] = (matrix[i][j] * matrix[i][j]) % mod
        exp //= 2

    return result

def solve(N: int, M: int, K: int, edges: List[List[int]]) -> int:
    """
    Solves the problem of finding the number of ways to move exactly K times in a directed graph.
    """
    # Create adjacency matrix
    adj_matrix = [[0] * N for _ in range(N)]
    for i in range(N):
        adj_matrix[i][(i + 1) % N] = 1
    for x, y in edges:
        adj_matrix[x - 1][y - 1] = 1

    # Compute the Kth power of the adjacency matrix modulo MOD
    adj_matrix_k = matrix_mod_exp(adj_matrix, K, MOD)

    # The answer is the value in the first row and first column of the resulting matrix
    return adj_matrix_k[0][0]

def main():
    N, M, K = map(int, input().split())
    edges = [list(map(int, input().split())) for _ in range(M)]
    print(solve(N, M, K, edges))

if __name__ == "__main__":
    main()