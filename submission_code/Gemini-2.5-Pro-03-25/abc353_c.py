# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read N from input, the number of elements in the sequence
    N = int(sys.stdin.readline())
    
    # Read the sequence A of N positive integers from input
    A = list(map(int, sys.stdin.readline().split()))
    
    # Define the modulus M = 10^8 as specified in the problem
    M = 100000000
    
    # Calculate the sum of all elements in A.
    # Python's integers support arbitrary precision, so intermediate large sums are handled correctly.
    sum_A = sum(A)
    
    # Calculate S_naive = sum_{1 <= i < j <= N} (A_i + A_j).
    # We can show that each element A_k appears exactly (N-1) times in this sum.
    # For a fixed k, A_k appears as the first term A_i in pairs (A_k, A_j) where j > k. There are N-k such pairs.
    # It appears as the second term A_j in pairs (A_i, A_k) where i < k. There are k-1 such pairs.
    # The total number of times A_k appears is (N-k) + (k-1) = N-1.
    # Therefore, S_naive = sum_{k=1}^N (N-1) * A_k = (N - 1) * sum(A).
    S_naive = (N - 1) * sum_A
    
    # The problem asks for S = sum_{1 <= i < j <= N} f(A_i, A_j), where f(x, y) = (x + y) % M.
    # We know that (x + y) % M = x + y - M * floor((x + y) / M).
    # So, S = sum_{i<j} (A_i + A_j - M * floor((A_i + A_j) / M))
    # S = sum_{i<j} (A_i + A_j) - M * sum_{i<j} floor((A_i + A_j) / M)
    # S = S_naive - M * K, where K = sum_{i<j} floor((A_i + A_j) / M).
    # Since 1 <= A_i < M, we have 2 <= A_i + A_j < 2M.
    # Thus, floor((A_i + A_j) / M) is 1 if A_i + A_j >= M, and 0 otherwise.
    # So K is the count of pairs (i, j) with i < j such that A_i + A_j >= M.

    # To efficiently calculate K, we first sort the array A. This takes O(N log N) time.
    A_sorted = sorted(A)
    
    # Now we use the two-pointer technique on the sorted array A_sorted to find K. This takes O(N) time.
    K = 0  # Initialize the count K to 0
    left = 0  # Initialize the left pointer to the start of the array
    right = N - 1  # Initialize the right pointer to the end of the array
    
    # Iterate while the left pointer is strictly less than the right pointer
    while left < right:
        # Check if the sum of elements at the current pointers meets the condition A_i + A_j >= M
        if A_sorted[left] + A_sorted[right] >= M:
            # If the condition A_sorted[left] + A_sorted[right] >= M is met:
            # The pair (A_sorted[left], A_sorted[right]) satisfies the condition.
            # Furthermore, because the array is sorted, any element A_sorted[k] with left <= k < right
            # will satisfy A_sorted[k] >= A_sorted[left].
            # Therefore, A_sorted[k] + A_sorted[right] >= A_sorted[left] + A_sorted[right] >= M.
            # This means all pairs (A_sorted[k], A_sorted[right]) for k from left to right-1 satisfy the condition.
            # The number of such indices k is (right - 1) - left + 1 = right - left.
            # We add this count to K.
            K += (right - left)
            # Since we have accounted for all pairs involving A_sorted[right] that satisfy the condition
            # (when paired with elements at index >= left), we can move the right pointer one step inward
            # to consider pairs with A_sorted[right-1].
            right -= 1
        else:
            # If the condition A_sorted[left] + A_sorted[right] < M is met:
            # The pair (A_sorted[left], A_sorted[right]) does not satisfy the condition.
            # Since A_sorted[right] is the largest element potentially pairable with A_sorted[left]
            # within the current scope of the right pointer (indices j >= left+1), and the sum is still less than M,
            # no element A_sorted[j] with j <= right can make the sum A_sorted[left] + A_sorted[j] >= M.
            # This is because A_sorted[j] <= A_sorted[right] for j <= right.
            # Therefore, to potentially satisfy the condition, we must use a larger element than A_sorted[left].
            # We move the left pointer one step inward (to the right).
            left += 1
            
    # Calculate the final answer S = S_naive - M * K
    # S_naive and M * K can be large, but Python handles large integers automatically.
    # The final result S = sum_{i<j} f(A_i, A_j) is guaranteed to be non-negative since f(x, y) >= 0.
    S = S_naive - M * K
    
    # Print the final answer to standard output
    print(S)

# Ensure the script runs the solve function when executed.
# This is standard practice for competitive programming submissions in Python.
if __name__ == '__main__':
    solve()