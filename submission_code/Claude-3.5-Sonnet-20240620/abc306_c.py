# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# Create a dictionary to store the indices of each number
indices = {i: [] for i in range(1, N+1)}

# Populate the indices dictionary
for i, num in enumerate(A, 1):
    indices[num].append(i)

# Calculate f(i) for each number
f = {i: indices[i][1] for i in range(1, N+1)}

# Sort the numbers based on f(i)
sorted_numbers = sorted(range(1, N+1), key=lambda x: f[x])

# Print the result
print(*sorted_numbers)