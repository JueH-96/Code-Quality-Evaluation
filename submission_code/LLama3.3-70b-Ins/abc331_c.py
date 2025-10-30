import sys

def solve():
    # Read input from stdin
    N = int(input())
    A = list(map(int, input().split()))

    # Initialize result list
    B = [0] * N

    # Calculate sum of elements greater than A_i for each i
    for i in range(N):
        for j in range(N):
            if A[j] > A[i]:
                B[i] += A[j]

    # Print result
    print(*B)

if __name__ == "__main__":
    solve()