import sys

def calculate_expression(N, A):
    """
    Calculate the given expression.

    Args:
    N (int): The number of elements in the sequence.
    A (list): The sequence of integers.

    Returns:
    int: The value of the expression.
    """
    total = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate max(A_j - A_i, 0) and add it to the total
            total += max(A[j] - A[i], 0)
    return total

def main():
    # Read the input from stdin
    N = int(input())
    A = list(map(int, input().split()))

    # Calculate and print the result
    result = calculate_expression(N, A)
    print(result)

if __name__ == "__main__":
    main()