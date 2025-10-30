def count_arithmetic_progressions(n, arr):
    """
    Count the number of pairs of integers (l, r) satisfying 1 <= l <= r <= n
    such that the subsequence (A_l, A_{l+1}, ..., A_r) forms an arithmetic progression.

    Args:
        n (int): The number of elements in the array.
        arr (list): The input array.

    Returns:
        int: The number of pairs of integers (l, r) that satisfy the condition.
    """
    count = 0
    for l in range(n):
        for r in range(l, n):
            # Check if the subsequence (A_l, A_{l+1}, ..., A_r) is an arithmetic progression
            if is_arithmetic_progression(arr[l:r+1]):
                count += 1
    return count


def is_arithmetic_progression(arr):
    """
    Check if the given array is an arithmetic progression.

    Args:
        arr (list): The input array.

    Returns:
        bool: True if the array is an arithmetic progression, False otherwise.
    """
    if len(arr) == 1:
        return True
    diff = arr[1] - arr[0]
    for i in range(2, len(arr)):
        if arr[i] - arr[i-1] != diff:
            return False
    return True


# Read the input from stdin
n = int(input())
arr = list(map(int, input().split()))

# Count the number of pairs of integers (l, r) that satisfy the condition
count = count_arithmetic_progressions(n, arr)

# Write the answer to stdout
print(count)