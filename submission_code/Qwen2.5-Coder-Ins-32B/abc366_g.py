# YOUR CODE HERE
import sys
import numpy as np

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    edges = []
    for i in range(M):
        u = int(input[2 + 2 * i]) - 1
        v = int(input[3 + 2 * i]) - 1
        edges.append((u, v))

    # Create the adjacency matrix
    adj_matrix = np.zeros((N, N), dtype=int)
    for u, v in edges:
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1

    # Create the system of linear equations in GF(2)
    A = np.zeros((N, N), dtype=int)
    b = np.zeros(N, dtype=int)
    for i in range(N):
        neighbors = np.where(adj_matrix[i] == 1)[0]
        for neighbor in neighbors:
            A[i, neighbor] = 1
        b[i] = 0

    # Solve the system of linear equations
    try:
        # Use numpy's linear algebra solver
        x = np.linalg.solve(A % 2, b % 2)
        # If a solution exists, we can assign any non-zero value to the vertices
        # Here we assign 2^i to each vertex i
        solution = [2**i for i in range(N)]
        print("Yes")
        print(" ".join(map(str, solution)))
    except np.linalg.LinAlgError:
        # If the system is singular, no solution exists
        print("No")

if __name__ == "__main__":
    main()