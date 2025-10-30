n = int(input())
a = list(map(int, input().split()))
head = None
for i in range(1, n+1):
    if a[i-1] == -1:
        head = i
        break
next_node = [None] * (n + 1)
for i in range(1, n+1):
    if a[i-1] != -1:
        next_node[a[i-1]] = i
result = []
current = head
while current is not None:
    result.append(current)
    current = next_node[current]
print(' '.join(map(str, result)))