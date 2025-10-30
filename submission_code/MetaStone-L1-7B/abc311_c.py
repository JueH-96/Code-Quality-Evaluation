n = int(input())
A = list(map(int, input().split()))
A = [0] + A  # Convert to 1-based indexing

# Phase 1: Find a meeting point using Floyd's Tortoise and Hare algorithm
slow = A[1]
fast = A[A[1]]
while slow != fast:
    slow = A[slow]
    fast = A[A[fast]]

# Phase 2: Find the entrance to the cycle
slow = 1
while slow != fast:
    slow = A[slow]
    fast = A[fast]

# Collect the cycle
current = slow
path = []
while True:
    path.append(current)
    current = A[current]
    if current == slow:
        break

print(len(path), end='')
print(' '.join(map(str, path)))