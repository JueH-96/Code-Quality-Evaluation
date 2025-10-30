import sys

def solve():
    # Read input from stdin
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    X = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

    # Calculate total units of each nutrient
    total_units = [0] * M
    for i in range(N):
        for j in range(M):
            total_units[j] += X[i][j]

    # Check if goal is met for all nutrients
    for i in range(M):
        if total_units[i] < A[i]:
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    solve()