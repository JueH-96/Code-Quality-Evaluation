import sys

def calculate_f(x, y):
    """Calculate the remainder of (x + y) divided by 10^8."""
    return (x + y) % (10**8)

def main():
    # Read input from stdin
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    # Initialize sum
    total_sum = 0

    # Calculate the sum of f(A_i, A_j) for all pairs (i, j) where i < j
    for i in range(N-1):
        for j in range(i+1, N):
            total_sum += calculate_f(A[i], A[j])

    # Print the result
    print(total_sum)

if __name__ == "__main__":
    main()