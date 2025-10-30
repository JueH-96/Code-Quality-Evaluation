import sys

def solve():
    N, A, B, C = map(int, sys.stdin.readline().split())
    D = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

    # Initialize the distance matrix
    for i in range(N):
        for j in range(N):
            if i == j:
                D[i][j] = 0

    # Fill in the distance matrix
    for k in range(N):
        for i in range(N):
            for j in range(N):
                D[i][j] = min(D[i][j], D[i][k] + D[k][j])

    # Calculate the minimum time
    min_time = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            time = D[i][j]*A + D[i][j]*B + C
            min_time = min(min_time, time)

    print(min_time)

if __name__ == "__main__":
    solve()