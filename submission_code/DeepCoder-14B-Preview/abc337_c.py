n = int(input())
A = list(map(int, input().split()))

# Find the head of the line
head = None
for i in range(1, n + 1):
    if A[i - 1] == -1:
        head = i
        break

# Create the next_person array
next_person = [0] * (n + 1)  # Using list for faster access
for i in range(1, n + 1):
    a_i = A[i - 1]
    if a_i != -1:
        next_person[a_i] = i

# Build the result by traversing from the head
result = []
current = head
result.append(current)
current = next_person[current]

while current != 0:
    result.append(current)
    current = next_person[current]

print(' '.join(map(str, result)))