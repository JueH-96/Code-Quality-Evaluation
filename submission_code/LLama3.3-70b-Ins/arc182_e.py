import sys

def solve(N, M, C, K, A):
    """
    Calculate the sum of minimum values of (Ck + Ai) mod M for k = 0 to K-1.

    Args:
    N (int): The number of elements in sequence A.
    M (int): The modulus value.
    C (int): The coefficient of k.
    K (int): The upper limit of k.
    A (list): The sequence of integers.

    Returns:
    int: The sum of minimum values.
    """
    total = 0
    for k in range(K):
        # Calculate (Ck + Ai) mod M for each Ai and find the minimum value
        min_val = min((C * k + Ai) % M for Ai in A)
        total += min_val
    return total

def main():
    # Read input from stdin
    N, M, C, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Solve the problem
    result = solve(N, M, C, K, A)

    # Write the answer to stdout
    print(result)

if __name__ == "__main__":
    main()