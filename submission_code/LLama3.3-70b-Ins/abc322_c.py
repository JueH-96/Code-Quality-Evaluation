import sys

def solve():
    # Read input from stdin
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize result array
    result = [0] * N

    # Iterate over each day
    for i in range(N):
        # Find the first day when fireworks are launched on or after the current day
        for j in range(M):
            if A[j] >= i + 1:
                # Calculate the number of days until fireworks are launched
                result[i] = A[j] - (i + 1)
                break

    # Print the result
    for i in range(N):
        print(result[i])

# Call the solve function
solve()