def max_kagamimochi(N, A):
    # Initialize two pointers
    left = 0
    right = 0
    pairs = 0
    
    # Use a two-pointer technique to find valid pairs
    while left < N and right < N:
        # Move the right pointer to find a valid mochi for the left mochi
        while right < N and (A[left] * 2 > A[right]):
            right += 1
        
        # If we found a valid pair
        if right < N:
            pairs += 1
            left += 1  # Move to the next mochi
            right += 1  # Use the current right mochi
        else:
            break  # No more valid pairs can be formed
    
    return pairs

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Calculate and print the result
result = max_kagamimochi(N, A)
print(result)