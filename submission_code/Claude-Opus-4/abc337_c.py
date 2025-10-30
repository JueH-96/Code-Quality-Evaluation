# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Find who comes after each person
next_person = {}
front = -1

for i in range(n):
    if a[i] == -1:
        front = i + 1  # person numbers are 1-indexed
    else:
        # person (i+1) is right behind person a[i]
        # so person a[i]'s next person is (i+1)
        next_person[a[i]] = i + 1

# Build the line starting from the front
result = []
current = front
while current is not None:
    result.append(current)
    current = next_person.get(current)

print(' '.join(map(str, result)))