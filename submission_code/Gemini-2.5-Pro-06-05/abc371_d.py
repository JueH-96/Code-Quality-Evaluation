# YOUR CODE HERE
import sys
from bisect import bisect_left, bisect_right

def solve():
    """
    Solves the village population range query problem.
    """
    # Use a faster I/O method for large inputs
    input = sys.stdin.readline

    # Read the number of villages
    try:
        N = int(input())
    except (ValueError, IndexError):
        # Handle empty input at the end of a file
        return
        
    # Read the sorted village coordinates
    X = list(map(int, input().split()))

    # Read the village populations
    P = list(map(int, input().split()))

    # --- Precomputation Step ---
    # Create a prefix sum array for the populations.
    # prefix_P[i] will store the sum of populations for villages from index 0 to i-1.
    # The array has size N+1 for easier calculation, with prefix_P[0] = 0.
    prefix_P = [0] * (N + 1)
    for i in range(N):
        prefix_P[i + 1] = prefix_P[i] + P[i]

    # Read the number of queries
    try:
        Q = int(input())
    except (ValueError, IndexError):
        return

    # --- Query Processing Step ---
    # Store answers in a list to print them all at once at the end.
    # This is generally faster than multiple print calls due to I/O buffering.
    answers = []
    for _ in range(Q):
        # Read the query range [L, R]
        L, R = map(int, input().split())

        # Find the index of the first village with coordinate >= L.
        # This is the starting index (inclusive) of our sum.
        start_index = bisect_left(X, L)

        # Find the index of the first village with coordinate > R.
        # The villages included in the query are those with indices from
        # start_index up to (but not including) this end_index.
        end_index = bisect_right(X, R)

        # Calculate the sum of populations for villages in the index range
        # [start_index, end_index - 1] in O(1) using the prefix sum array.
        total_villagers = prefix_P[end_index] - prefix_P[start_index]

        answers.append(str(total_villagers))

    # Print all collected answers, each on a new line.
    print("
".join(answers))

# Run the solution
solve()