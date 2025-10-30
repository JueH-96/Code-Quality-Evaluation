def max_slime_sizes(N, A):
    results = [0] * N
    
    # For each slime, we will calculate the maximum size it can achieve
    for k in range(N):
        current_size = A[k]
        left_index = k - 1
        right_index = k + 1
        
        # Absorb slimes to the left
        while left_index >= 0 and A[left_index] < current_size:
            current_size += A[left_index]
            left_index -= 1
        
        # Reset current size for right absorption
        current_size = A[k]
        
        # Absorb slimes to the right
        while right_index < N and A[right_index] < current_size:
            current_size += A[right_index]
            right_index += 1
        
        results[k] = current_size
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Get the results
results = max_slime_sizes(N, A)

# Print the results
print(" ".join(map(str, results)))