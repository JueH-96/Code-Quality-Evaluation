# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

next_person = [0] * (n + 1)
for i in range(n):
    next_person[i + 1] = a[i]

start = -1
for i in range(1, n + 1):
    if next_person[i] == -1:
        start = i
        break

result = []
current = start
while current != 0:
    result.append(current)
    current = next_person[current]

print(*result)