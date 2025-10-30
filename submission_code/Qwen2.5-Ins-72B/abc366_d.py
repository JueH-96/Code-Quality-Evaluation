# YOUR CODE HERE
import sys

def read_input():
    N = int(input())
    A = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            row = list(map(int, input().split()))
            for z in range(1, N+1):
                A[x][y][z] = row[z-1]
    
    Q = int(input())
    queries = []
    for _ in range(Q):
        queries.append(list(map(int, input().split())))
    return N, A, Q, queries

def preprocess(A, N):
    # Precompute the prefix sums
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                A[x][y][z] += A[x-1][y][z] + A[x][y-1][z] + A[x][y][z-1] - A[x-1][y-1][z] - A[x-1][y][z-1] - A[x][y-1][z-1] + A[x-1][y-1][z-1]

def solve_query(A, Lx, Rx, Ly, Ry, Lz, Rz):
    return A[Rx][Ry][Rz] - A[Lx-1][Ry][Rz] - A[Rx][Ly-1][Rz] - A[Rx][Ry][Lz-1] + A[Lx-1][Ly-1][Rz] + A[Lx-1][Ry][Lz-1] + A[Rx][Ly-1][Lz-1] - A[Lx-1][Ly-1][Lz-1]

def main():
    N, A, Q, queries = read_input()
    preprocess(A, N)
    for query in queries:
        Lx, Rx, Ly, Ry, Lz, Rz = query
        print(solve_query(A, Lx, Rx, Ly, Ry, Lz, Rz))

if __name__ == "__main__":
    main()