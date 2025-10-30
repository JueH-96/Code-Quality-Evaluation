import sys

def calculate_f(A, K):
    """
    Calculate the function f(A) as defined in the problem statement.

    Args:
    A (list): The input sequence.
    K (int): The number of terms to sum.

    Returns:
    int: The value of f(A).
    """
    # Sort the sequence in descending order
    B = sorted(A, reverse=True)
    # Sum the first K terms
    return sum(B[:K])

def main():
    # Read the input from stdin
    N, K, Q = map(int, sys.stdin.readline().split())
    A = [0] * N

    for _ in range(Q):
        X, Y = map(int, sys.stdin.readline().split())
        # Update the sequence
        A[X - 1] = Y
        # Calculate and print the value of f(A)
        print(calculate_f(A, K))

if __name__ == "__main__":
    main()