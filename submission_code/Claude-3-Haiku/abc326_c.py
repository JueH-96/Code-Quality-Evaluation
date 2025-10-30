# Read the input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Sort the coordinates
A.sort()

# Initialize the maximum number of gifts
max_gifts = 0

# Iterate through the coordinates
for i in range(N):
    # Find the index of the first gift that is outside the interval
    j = i
    while j < N and A[j] < A[i] + M:
        j += 1
    
    # Update the maximum number of gifts
    max_gifts = max(max_gifts, j - i)

# Print the answer
print(max_gifts)