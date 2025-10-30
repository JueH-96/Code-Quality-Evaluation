import sys

# Read input faster for large inputs
# Using sys.stdin.readline for potentially faster I/O compared to input()
input = sys.stdin.readline

def solve():
    """
    Solves the problem of finding the sum of integers between 1 and K (inclusive)
    that do not appear in the sequence A.
    """
    # Read N and K from the first line of input
    # N is the length of the sequence A
    # K is the upper bound of the range [1, K]
    n, k = map(int, input().split())

    # Read the sequence A from the second line of input
    # We only need the unique elements of A that are within the range [1, K].
    # We can process the input line directly to build this set efficiently.

    # Create an iterator that yields integers from the input line
    a_iterator = map(int, input().split())

    # Filter the iterator to keep only elements x such that 1 <= x <= K
    a_in_range_iterator = filter(lambda x: 1 <= x <= k, a_iterator)

    # Create a set from the filtered iterator to get unique elements in A that are <= K
    unique_a_in_range = set(a_in_range_iterator)

    # Calculate the sum of all integers from 1 to K using the arithmetic series formula.
    # Formula: sum = k * (k + 1) / 2
    # Python's integers handle arbitrarily large numbers, so overflow is not an issue here
    # even for K up to 2 * 10^9.
    total_sum_1_to_k = k * (k + 1) // 2

    # Calculate the sum of the unique elements from A that are within the range [1, K].
    # sum() applied to a set efficiently calculates the sum of its elements.
    sum_a_in_range = sum(unique_a_in_range)

    # The sum of integers from 1 to K that are *not* in A is the total sum of integers
    # from 1 to K minus the sum of the unique elements from A that fall within this range.
    result = total_sum_1_to_k - sum_a_in_range

    # Print the final calculated result to standard output.
    print(result)

# Call the solve function to execute the logic when the script is run.
solve()