# YOUR CODE HERE
import sys

def main():
    """
    Solves the problem by pre-computing distinct counts for all prefixes and suffixes.
    """
    try:
        # Read input from stdin for efficiency
        # N: the number of elements in the sequence
        N = int(sys.stdin.readline())
        # A: the sequence of integers
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Gracefully exit on malformed or empty input
        return

    # --- Step 1: Pre-compute distinct counts for all prefixes ---
    # left_distinct[i] will store the number of distinct elements in the subarray A[0...i].
    # We use 0-based indexing for the array A.
    left_distinct = [0] * N
    seen = set()
    for i in range(N):
        seen.add(A[i])
        left_distinct[i] = len(seen)

    # --- Step 2: Pre-compute distinct counts for all suffixes ---
    # right_distinct[i] will store the number of distinct elements in the subarray A[i...N-1].
    right_distinct = [0] * N
    seen = set()
    for i in range(N - 1, -1, -1):
        seen.add(A[i])
        right_distinct[i] = len(seen)

    # --- Step 3: Find the maximum sum over all possible splits ---
    # A split is made after index `i`, where 0 <= i <= N-2.
    # This corresponds to the problem statement's split point `i+1` (1-indexed).
    # The left part is A[0...i] and the right part is A[i+1...N-1].
    max_sum = 0
    for i in range(N - 1):
        # The number of distinct elements in the left part is left_distinct[i].
        # The number of distinct elements in the right part is right_distinct[i+1].
        current_sum = left_distinct[i] + right_distinct[i + 1]
        
        # Update the overall maximum sum.
        if current_sum > max_sum:
            max_sum = current_sum

    # Print the final answer to stdout.
    print(max_sum)

if __name__ == "__main__":
    main()