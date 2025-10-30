N = int(input())
A = []
for i in range(N):
    row = list(map(int, input().split()))
    A.append(row)

current = 1
for i in range(N):
    if current <= i:
        current = A[i][current-1]
    else:
        current = A[current-1][i]

print(current)