import sys

def readints():
    return list(map(int, sys.stdin.readline().split()))

def main():
    N, M = readints()
    adj = [[] for _ in range(N+1)]  # vertices are 1-based
    for _ in range(M):
        u, v = readints()
        adj[u].append(v)
        adj[v].append(u)
    
    # For each bit, create the system of equations
    bit_solutions = []
    for k in range(60):
        matrix = []
        for v in range(1, N+1):
            if len(adj[v]) == 0:
                # No equation for this vertex
                continue
            row = 0
            for u in adj[v]:
                row |= 1 << (u-1)  # u is 1-based, u-1 is 0-based
            matrix.append(row)
        # Perform Gaussian elimination
        solution = gauss(matrix, N)
        bit_solutions.append(solution)
    
    # Check each vertex's X_v is non-zero
    valid = True
    for v in range(1, N+1):
        X_v = 0
        for k in range(60):
            X_v += (bit_solutions[k][v-1] << k)
        if X_v == 0:
            valid = False
            break
    if not valid:
        print("No")
        return
    
    # Output the solution
    print("Yes")
    X = [0] * N
    for k in range(60):
        for v in range(N):
            X[v] += (bit_solutions[k][v] << k)
    for x in X:
        print(x)
    
def gauss(matrix, n_vars):
    rank = 0
    for col in range(n_vars):
        pivot = -1
        for r in range(rank, len(matrix)):
            if (matrix[r] >> col) & 1:
                pivot = r
                break
        if pivot == -1:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        for r in range(len(matrix)):
            if r != rank and ((matrix[r] >> col) & 1):
                matrix[r] ^= matrix[rank]
        rank += 1
    solution = [0] * n_vars
    for r in range(rank):
        pivot_col = -1
        for c in range(n_vars):
            if (matrix[r] >> c) & 1:
                pivot_col = c
                break
        if pivot_col == -1:
            continue
        solution[pivot_col] = (matrix[r] >> pivot_col) & 1
    return solution

if __name__ == '__main__':
    main()