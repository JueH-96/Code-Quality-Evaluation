def max_pyramid_sequence_size(N, A):
    # We will use a two-pointer technique to find the maximum size of the pyramid sequence
    left = 0
    right = N - 1
    max_k = 0
    
    # We will check for the maximum k such that we can form the pyramid sequence of size k
    while left <= right:
        # Calculate the current k
        k = (right - left + 1) // 2 + 1
        
        # Check if we can form a pyramid sequence of size k
        can_form = True
        for i in range(k):
            # The expected value at position i in the pyramid sequence is k - i
            expected_value = k - i
            # Check if we can satisfy the pyramid sequence condition
            if A[left + i] < expected_value:
                can_form = False
                break
            if A[right - i] < expected_value:
                can_form = False
                break
        
        if can_form:
            max_k = k  # We can form a pyramid of size k
            left += 1  # Try to increase k
        else:
            right -= 1  # Decrease the size of the sequence
    
    return max_k

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Get the result
result = max_pyramid_sequence_size(N, A)

# Print the result
print(result)