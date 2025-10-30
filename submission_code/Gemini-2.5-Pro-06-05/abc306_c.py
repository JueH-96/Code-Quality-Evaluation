import sys

# YOUR CODE HERE
def solve():
    """
    This function solves the problem by finding the middle occurrence index for each number
    and then sorting the numbers based on these indices using a linear time O(N) algorithm.
    """
    
    # Read N from standard input
    try:
        n_str = sys.stdin.readline()
        if not n_str: return
        N = int(n_str)
        
        a_str = sys.stdin.readline()
        if not a_str: return
        A = list(map(int, a_str.split()))
    except (IOError, ValueError):
        # Gracefully handle empty or malformed input
        return

    # Step 1: Find f(i) for all i in a single pass.
    # `counts[i]` tracks how many times number `i` has been seen.
    # `f_values[i]` will store the index of the middle occurrence of number `i`.
    counts = [0] * (N + 1)
    f_values = [0] * (N + 1)

    # Iterate through the input array A to find the middle occurrence of each number.
    for idx, num in enumerate(A):
        counts[num] += 1
        # When a number is seen for the second time, we've found its middle occurrence.
        if counts[num] == 2:
            # The problem uses 1-based indexing for positions, so we use idx + 1.
            f_values[num] = idx + 1

    # Step 2: Sort the numbers 1...N based on their f_values in linear time.
    # We use a list as a map where the index represents the f-value.
    # The size must be 3*N + 1 to accommodate the largest possible index.
    sorted_by_f = [0] * (3 * N + 1)

    # Populate this map: `sorted_by_f[f(i)] = i`
    for i in range(1, N + 1):
        f_i = f_values[i]
        sorted_by_f[f_i] = i

    # Step 3: Collect the results.
    # Iterate through `sorted_by_f`. The non-zero elements are the numbers 1...N,
    # now sorted by their f-values because the indices of `sorted_by_f` are ordered.
    result = [num for num in sorted_by_f if num > 0]

    # Step 4: Print the final sorted sequence.
    print(*result)

solve()