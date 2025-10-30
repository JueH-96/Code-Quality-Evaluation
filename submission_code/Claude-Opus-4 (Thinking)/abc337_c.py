# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Find who is behind each person
behind = {}
for i in range(n):
    if a[i] != -1:
        behind[a[i] - 1] = i  # Convert to 0-indexed

# Find the front person
front = -1
for i in range(n):
    if a[i] == -1:
        front = i
        break

# Build the order by traversing the chain
order = []
current = front
while current != -1:
    order.append(current + 1)  # Convert back to 1-indexed
    current = behind.get(current, -1)

print(' '.join(map(str, order)))