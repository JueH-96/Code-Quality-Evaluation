# YOUR CODE HERE
import sys

# Function to perform ceiling division for positive denominator
def ceil_div(a, b):
    # This formula (a + b - 1) // b works for any integer 'a' and positive integer 'b'.
    # It correctly computes ceil(a/b) using integer arithmetic (floor division).
    return (a + b - 1) // b

def solve():
    A, M, L, R = map(int, sys.stdin.readline().split())

    # We are looking for integers k such that L <= A + k*M <= R
    # This can be rewritten as:
    # L - A <= k*M <= R - A

    # Let lower_bound_for_k_M = L - A
    # Let upper_bound_for_k_M = R - A
    lower_bound_for_k_M = L - A
    upper_bound_for_k_M = R - A

    # Now we need to find k such that lower_bound_for_k_M <= k*M <= upper_bound_for_k_M
    # Since M > 0 (as per constraints, 1 <= M), we can divide by M:
    # (lower_bound_for_k_M) / M <= k <= (upper_bound_for_k_M) / M

    # The smallest integer k (k_start) is ceil((L - A) / M)
    k_start = ceil_div(lower_bound_for_k_M, M)

    # The largest integer k (k_end) is floor((R - A) / M)
    # Python's // operator is floor division, so it works directly for k_end
    k_end = upper_bound_for_k_M // M

    # The number of integers k in the range [k_start, k_end] (inclusive) is k_end - k_start + 1.
    # If k_start > k_end, it means there are no such integers, so the count is 0.
    count = max(0, k_end - k_start + 1)

    print(count)

# Call the solve function to run the program
solve()