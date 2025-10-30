# YOUR CODE HERE
import sys

# Function to read input efficiently using sys.stdin.readline
def get_ints():
    """Reads a line from stdin, splits it by spaces, and maps the parts to integers."""
    return map(int, sys.stdin.readline().strip().split())

def solve():
    """Solves the problem based on the input."""
    # Read N, the number of elements in the sequence
    n, = get_ints()
    # Read the sequence A
    a = list(get_ints())

    # Determine the maximum possible value for elements in A based on constraints.
    # Constraints state 1 <= A_i <= 1,000,000.
    M = 1000000

    # Create an array to store the sum of contributions for each distinct value.
    # sum_contrib[x] will store the sum of all elements in A that are equal to x.
    # The size needs to be M + 1 to accommodate index M. Indices range from 0 to M.
    # Since constraints state A_i >= 1, index 0 will naturally remain 0.
    sum_contrib = [0] * (M + 1)
    for x in a:
        # According to constraints, 1 <= x <= M, so x is always a valid index.
        sum_contrib[x] += x

    # Create an array to store the suffix sum of contributions.
    # suffix_sum_gt[val] will store the sum of all elements A_j in the original sequence
    # such that A_j > val.
    # To calculate suffix_sum_gt[val], we need access to suffix_sum_gt[val + 1]
    # and sum_contrib[val + 1]. If val can go up to M, we need index M+1.
    # Therefore, the size of suffix_sum_gt should be M + 2 to allow access to index M+1.
    # suffix_sum_gt[M+1] will be initialized to 0, representing the sum of elements > M+1 (which is 0).
    suffix_sum_gt = [0] * (M + 2)

    # Calculate the suffix sums efficiently by iterating downwards.
    # The loop iterates from val = M down to val = 0.
    # For each val, suffix_sum_gt[val] is computed based on the previously computed
    # suffix_sum_gt[val + 1] and the contribution of the value val + 1.
    # The formula suffix_sum_gt[val] = suffix_sum_gt[val + 1] + sum_contrib[val + 1]
    # correctly accumulates the sum of contributions for all values strictly greater than val.
    for val in range(M, -1, -1):
        suffix_sum_gt[val] = suffix_sum_gt[val + 1] + sum_contrib[val + 1]

    # Build the result list.
    # For each element a[i] in the original sequence A, the required sum is the sum
    # of all elements A_j such that A_j > a[i]. This value is precomputed and stored
    # in suffix_sum_gt[a[i]].
    # We use a list comprehension for conciseness and potential minor performance benefit.
    results = [suffix_sum_gt[a[i]] for i in range(n)]

    # Print the computed results for each A_i, separated by spaces.
    # The '*' operator unpacks the list 'results' into separate arguments for print.
    print(*results)

# Call the solve function to execute the main logic when the script is run.
if __name__ == '__main__':
    solve()