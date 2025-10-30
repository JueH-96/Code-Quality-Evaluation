import sys

# Function to calculate the sum of distinct values over all contiguous subsegments
def solve():
    # Read the number of elements N
    N = int(sys.stdin.readline())

    # Read the sequence of integers A. Store as a 0-indexed list.
    # A_k in the problem statement (1-based index) corresponds to A_list[k-1] (0-based index).
    A_list = list(map(int, sys.stdin.readline().split()))

    # last_seen array stores the 0-based index of the last occurrence for each value.
    # The values in A range from 1 to N. We use an array of size N+1 to map values to indices.
    # Initialize with -1 to indicate that no occurrence has been seen yet for any value.
    # last_seen[v] = -1 if value v has not appeared before index 0.
    last_seen = [-1] * (N + 1)

    # total_sum will store the sum of f(i, j) over all 0 <= i <= j <= N-1.
    # Initialize the total sum to 0.
    total_sum = 0

    # Iterate through the array using 0-based index k from 0 to N-1.
    # A_list[k] is the current element being processed.
    for k in range(N):
        # Get the value of the current element A_list[k].
        value = A_list[k]

        # Get the 0-based index of the previous occurrence of 'value'.
        # If this is the first time we see 'value' (in the entire array up to k),
        # last_seen[value] is still -1, which is correct for prev_k.
        prev_k = last_seen[value]

        # The element A_list[k] contributes 1 to the distinct count f(i, j)
        # for a subsegment A_list[i...j] if and only if:
        # 1. The index k is within the subsegment: i <= k <= j.
        # 2. A_list[k] is the first occurrence of its value in the subsegment A_list[i...j].
        #    This second condition is equivalent to saying that there is no index m
        #    such that i <= m < k and A_list[m] = A_list[k].
        #    This means that the largest index m < k where A_list[m] = A_list[k] (which is prev_k)
        #    must be strictly less than the starting index i of the subsegment.
        #    So, the conditions for A_list[k] to contribute 1 to f(i, j) are:
        #    0 <= i <= k <= j <= N-1 AND prev_k < i.

        # We need to count the number of pairs (i, j) that satisfy these conditions for the current k.
        # The possible values for i are integers such that prev_k < i <= k.
        # Since i must also be at least 0, the range for i is [max(0, prev_k + 1), k].
        # As prev_k is at least -1, prev_k + 1 is at least 0. So the range for i is [prev_k + 1, k].
        # The number of possible values for i is k - (prev_k + 1) + 1 = k - prev_k.

        # The possible values for j are integers such that k <= j <= N-1.
        # The number of possible values for j is (N-1) - k + 1 = N - k.

        # For a fixed k, the number of pairs (i, j) satisfying the conditions is the product
        # of the number of choices for i and the number of choices for j.
        # This is (k - prev_k) * (N - k).
        # This product represents the contribution of the element A_list[k] to the total sum.

        contribution = (k - prev_k) * (N - k)

        # Add this contribution to the total sum.
        total_sum += contribution

        # Update the last seen 0-based index for the current value 'value' to k.
        last_seen[value] = k

    # Print the final calculated total sum.
    # Python's integers handle large values automatically.
    print(total_sum)

# Check if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    # Call the solve function to run the main logic
    solve()