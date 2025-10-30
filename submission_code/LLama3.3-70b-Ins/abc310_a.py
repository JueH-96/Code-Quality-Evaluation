import sys

def solve():
    # Read input from stdin
    N, P, Q = map(int, sys.stdin.readline().split())
    D = list(map(int, sys.stdin.readline().split()))

    # Calculate the minimum total amount of money
    min_total = P
    for d in D:
        min_total = min(min_total, Q + d)

    # Print the answer
    print(min_total)

# Call the solve function
solve()