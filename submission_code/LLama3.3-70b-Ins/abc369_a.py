def count_arithmetic_sequences(A, B):
    """
    Counts the number of integers x that can be arranged with A and B to form an arithmetic sequence.

    Args:
        A (int): The first integer.
        B (int): The second integer.

    Returns:
        int: The number of integers x that satisfy the condition.
    """
    count = 0
    for x in range(-1000, 1001):  # Assuming x is within a reasonable range
        # Check all possible orders of A, B, and x
        for p, q, r in [(A, B, x), (A, x, B), (B, A, x), (B, x, A), (x, A, B), (x, B, A)]:
            if q - p == r - q:
                count += 1
                break  # No need to check other orders for the same x
    return count

# Read input from stdin
A, B = map(int, input().split())

# Print the result
print(count_arithmetic_sequences(A, B))