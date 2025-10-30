n = int(input())
a = list(map(int, input().split()))

# Find who is at the front
front = -1
for i in range(n):
    if a[i] == -1:
        front = i + 1  # Convert to 1-indexed
        break

# Build next mapping: who is behind each person
next_person = {}
for i in range(n):
    if a[i] != -1:
        next_person[a[i]] = i + 1  # Convert to 1-indexed

# Start from front and follow the chain
result = []
current = front
while current is not None:
    result.append(current)
    current = next_person.get(current)

print(' '.join(map(str, result)))