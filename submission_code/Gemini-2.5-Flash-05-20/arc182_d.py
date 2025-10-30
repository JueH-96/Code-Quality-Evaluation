import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    total_operations = 0

    for i in range(N):
        diff = abs(A[i] - B[i])
        # The shortest distance on a circle M
        cost = min(diff, M - diff)
        total_operations += cost

    print(total_operations)

# Ensure the script runs the solve function when executed
if __name__ == '__main__':
    solve()