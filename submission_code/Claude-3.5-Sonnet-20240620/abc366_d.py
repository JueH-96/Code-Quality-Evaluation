# YOUR CODE HERE
import sys

def read_input():
    N = int(sys.stdin.readline())
    A = [[[int(x) for x in sys.stdin.readline().split()] for _ in range(N)] for _ in range(N)]
    Q = int(sys.stdin.readline())
    queries = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    return N, A, Q, queries

def calculate_prefix_sum(A):
    N = len(A)
    prefix_sum = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
    
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                prefix_sum[x][y][z] = (
                    A[x-1][y-1][z-1] +
                    prefix_sum[x-1][y][z] +
                    prefix_sum[x][y-1][z] +
                    prefix_sum[x][y][z-1] -
                    prefix_sum[x-1][y-1][z] -
                    prefix_sum[x-1][y][z-1] -
                    prefix_sum[x][y-1][z-1] +
                    prefix_sum[x-1][y-1][z-1]
                )
    
    return prefix_sum

def query_sum(prefix_sum, Lx, Rx, Ly, Ry, Lz, Rz):
    return (
        prefix_sum[Rx][Ry][Rz] -
        prefix_sum[Lx-1][Ry][Rz] -
        prefix_sum[Rx][Ly-1][Rz] -
        prefix_sum[Rx][Ry][Lz-1] +
        prefix_sum[Lx-1][Ly-1][Rz] +
        prefix_sum[Lx-1][Ry][Lz-1] +
        prefix_sum[Rx][Ly-1][Lz-1] -
        prefix_sum[Lx-1][Ly-1][Lz-1]
    )

def main():
    N, A, Q, queries = read_input()
    prefix_sum = calculate_prefix_sum(A)
    
    for query in queries:
        Lx, Rx, Ly, Ry, Lz, Rz = query
        result = query_sum(prefix_sum, Lx, Rx, Ly, Ry, Lz, Rz)
        print(result)

if __name__ == "__main__":
    main()