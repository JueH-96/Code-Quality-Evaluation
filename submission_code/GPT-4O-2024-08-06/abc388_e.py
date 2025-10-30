def max_kagamimochi(N, A):
    i = 0
    j = 0
    count = 0
    
    while i < N and j < N:
        # Move j to find a valid pair with i
        while j < N and A[i] > A[j] / 2:
            j += 1
        # If j is valid, we found a pair
        if j < N:
            count += 1
            j += 1  # Move j to the next potential pair
        i += 1  # Move i to the next base mochi
    
    return count

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Output the result
print(max_kagamimochi(N, A))