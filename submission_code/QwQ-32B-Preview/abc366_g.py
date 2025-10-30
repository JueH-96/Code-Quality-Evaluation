def main():
    import sys
    from typing import List

    def gaussian_elimination(A: List[List[int]], N: int):
        from copy import deepcopy
        mat = deepcopy(A)
        rank = 0
        pivots = []
        for col in range(N):
            if rank >= N:
                break
            pivot = -1
            for row in range(rank, N):
                if mat[row][col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            mat[pivot], mat[rank] = mat[rank], mat[pivot]
            pivots.append((rank, col))
            for row in range(N):
                if row != rank and mat[row][col] == 1:
                    for col2 in range(N):
                        mat[row][col2] ^= mat[rank][col2]
            rank += 1
        return rank, pivots, mat

    def find_kernel(A: List[List[int]], N: int, rank: int, pivots: List[tuple]):
        kernel = []
        for free_var in range(N):
            is_free = True
            for pivot in pivots:
                if pivot[1] == free_var:
                    is_free = False
                    break
            if is_free:
                sol = [0] * N
                sol[free_var] = 1
                for pivot in pivots:
                    row, col = pivot
                    if A[row][free_var] == 1:
                        sol[col] = 1
                kernel.append(sol)
        return kernel

    input = sys.stdin.read().splitlines()
    N_M = input[0].split()
    N = int(N_M[0])
    M = int(N_M[1])
    edges = []
    for line in input[1:M+1]:
        u_v = line.split()
        u = int(u_v[0]) - 1  # Convert to 0-based index
        v = int(u_v[1]) - 1
        edges.append((u, v))

    # Build adjacency matrix
    A = [[0] * N for _ in range(N)]
    for u, v in edges:
        A[u][v] = 1
        A[v][u] = 1

    # Perform Gaussian elimination
    rank, pivots, _ = gaussian_elimination(A, N)

    if rank < N:
        # Find kernel basis
        kernel = find_kernel(A, N, rank, pivots)
        if not kernel:
            print("No")
            return
        # Choose the first basis vector as solution
        X = kernel[0]
        # Convert to decimal
        X_decimal = [str(x) for x in X]
        print("Yes")
        print(' '.join(X_decimal))
    else:
        print("No")

if __name__ == "__main__":
    main()