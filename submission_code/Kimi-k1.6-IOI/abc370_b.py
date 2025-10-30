n = int(input())
a = []
for _ in range(n):
    row = list(map(int, input().split()))
    a.append(row)
current = 1
for next_elem in range(1, n+1):
    i = current
    j = next_elem
    if i >= j:
        current = a[i-1][j-1]
    else:
        current = a[j-1][i-1]
print(current)