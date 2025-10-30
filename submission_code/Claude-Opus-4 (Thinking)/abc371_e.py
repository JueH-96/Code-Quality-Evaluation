# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Convert to 0-indexed for easier computation
# Find previous occurrence for each position
prev = [-1] * n
last_pos = {}
for i in range(n):
    if a[i] in last_pos:
        prev[i] = last_pos[a[i]]
    last_pos[a[i]] = i

# Calculate the answer
total = 0
for k in range(n):
    # Contribution of position k
    # Number of intervals [i,j] where a[k] is the first occurrence of its value
    total += (k - prev[k]) * (n - k)

print(total)