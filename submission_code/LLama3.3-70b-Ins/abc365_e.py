import sys

def calculate_xor_sum(N, A):
    """
    Calculate the sum of XOR of all possible subarrays.

    Args:
    N (int): The length of the input sequence.
    A (list): The input sequence.

    Returns:
    int: The sum of XOR of all possible subarrays.
    """
    total_sum = 0
    for i in range(N - 1):
        xor = 0
        for j in range(i, N):
            xor ^= A[j]
            if i != j:
                total_sum += xor
    return total_sum

def main():
    # Read input from stdin
    N = int(input())
    A = list(map(int, input().split()))

    # Calculate and print the result
    result = calculate_xor_sum(N, A)
    print(result)

if __name__ == "__main__":
    main()