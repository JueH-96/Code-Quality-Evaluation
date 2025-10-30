import sys

def solve():
    """
    Reads the input, solves the problem, and prints the output.
    """
    try:
        # Read input from stdin for efficiency
        n_str = sys.stdin.readline()
        if not n_str:
            return
        n = int(n_str)
        a = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Gracefully handle empty or malformed input
        return

    # The sum of elements is an invariant under the specified operation.
    total_sum = sum(a)

    # The target state is an array where all elements differ by at most 1.
    # This implies all elements must be either `base` or `base + 1`.
    # Let `num_high` be the count of `base + 1` and `num_low` be the count of `base`.
    # total_sum = num_low * base + num_high * (base + 1)
    # total_sum = (num_low + num_high) * base + num_high
    # total_sum = n * base + num_high
    # So, `base` is the quotient and `num_high` is the remainder.
    base = total_sum // n
    num_high = total_sum % n
    num_low = n - num_high

    # To minimize the number of operations (total value moved), we should
    # transform the smallest `num_low` elements of A into `base` and the
    # largest `num_high` elements of A into `base + 1`.
    # We sort A to achieve this optimal matching.
    a.sort()

    # The number of operations is the total value that needs to be added
    # to elements that are smaller than their targets. This is equal to
    # the total value that needs to be subtracted from elements that are
    # larger than their targets. We calculate the sum of increments.
    operations = 0
    
    # Calculate the cost for elements that need to become `base`.
    # These are the smallest `num_low` elements of the sorted array.
    target_low = base
    for i in range(num_low):
        if a[i] < target_low:
            operations += target_low - a[i]
    
    # Calculate the cost for elements that need to become `base + 1`.
    # These are the largest `num_high` elements of the sorted array.
    target_high = base + 1
    for i in range(num_low, n):
        if a[i] < target_high:
            operations += target_high - a[i]

    # Print the minimum number of operations required.
    print(operations)

solve()