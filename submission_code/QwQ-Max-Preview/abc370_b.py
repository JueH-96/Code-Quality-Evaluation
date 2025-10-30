n = int(input())
A = [[]]
for i in range(1, n+1):
    parts = list(map(int, input().split()))
    A.append(parts)
current = 1
for j in range(1, n+1):
    if current >= j:
        current = A[current][j-1]
    else:
        current = A[j][current-1]
print(current)