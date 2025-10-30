# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Create a mapping: person -> who is behind them
behind = {}
front = -1

for i in range(n):
    person = i + 1  # person numbers are 1-indexed
    if a[i] == -1:
        front = person
    else:
        behind[a[i]] = person

# Start from the front and follow the chain
result = []
current = front

while current != -1:
    result.append(current)
    current = behind.get(current, -1)

print(' '.join(map(str, result)))