# YOUR CODE HERE
import sys

# Using sys.stdin.readline for potentially faster I/O
input = sys.stdin.readline

def solve():
    # Read N, the length of the string
    N = int(input())
    # Read the string S consisting of '0's and '1's
    S = input().strip()

    # Store the 0-based indices of all '1's in the string S
    p_indices = []
    for i, char in enumerate(S):
        if char == '1':
            p_indices.append(i)

    # K is the total count of '1's in the string
    K = len(p_indices)
    
    # The problem statement guarantees that S contains at least one '1', so K >= 1.
    # Therefore, p_indices list is never empty.
    
    # If K == 1, there is only one '1'. It is already contiguous. The calculation below will result in 0 swaps.
    # If K == N, the string consists of all '1's. It is already contiguous. The calculation below will result in 0 swaps.
    # Thus, no special handling is needed for these edge cases.

    # We transform the problem based on the property that the minimum number of swaps
    # to make '1's contiguous starting at index `l` is given by Sum(|p_k - (l+k)|) for k=0..K-1.
    # Let x_k = p_k - k. Then the sum becomes Sum(|x_k - l|).
    # This sum is minimized when `l` is the median of the values {x_k}.
    
    # Calculate the transformed values x_k = p_k - k for each '1'
    # p_k is the original index of the k-th '1' (0-based index k)
    x_values = []
    for k in range(K):
        # k is the 0-based order index of the '1'
        x_values.append(p_indices[k] - k)

    # Sort the transformed values to easily find the median
    # Sorting takes O(K log K) time. Since K <= N, this is at most O(N log N).
    x_values.sort()
    
    # Find the median of the sorted x_values list.
    # For a list of length K, the median index (0-based) is (K-1)//2.
    # If K is odd, this picks the unique middle element.
    # If K is even, this picks the left of the two middle elements. Any value
    # in the interval between the two middle elements minimizes the sum of absolute differences.
    # So picking x_values[(K-1)//2] as the reference point `l` is valid.
    median_idx = (K - 1) // 2
    median_val = x_values[median_idx]
    
    # Compute the minimum total number of swaps required.
    # This equals the sum of absolute differences between each x_value
    # and the median value. This computation takes O(K) time.
    min_swaps = 0
    for val in x_values:
        min_swaps += abs(val - median_val)

    # Print the final minimum number of swaps
    print(min_swaps)

# Execute the main solution function
solve()