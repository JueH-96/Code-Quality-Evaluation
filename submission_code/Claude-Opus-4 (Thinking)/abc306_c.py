# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Dictionary to store all indices where each number appears
indices = {}
for i in range(3 * n):
    num = a[i]
    if num not in indices:
        indices[num] = []
    indices[num].append(i + 1)  # Convert to 1-indexed

# Calculate f(i) for each number i
f = {}
for i in range(1, n + 1):
    # The middle occurrence is the second element in the sorted list of indices
    f[i] = indices[i][1]

# Sort numbers 1 to N by their f values
numbers = list(range(1, n + 1))
numbers.sort(key=lambda x: f[x])

# Print the result
print(' '.join(map(str, numbers)))