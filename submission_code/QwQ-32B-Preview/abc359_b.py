N = int(input())
A = list(map(int, input().split()))
pos = {i: [] for i in range(1, N+1)}
for i in range(2*N):
    color = A[i]
    pos[color].append(i+1)
count = 0
for color in range(1, N+1):
    p1, p2 = pos[color]
    if abs(p1 - p2) == 2:
        count += 1
print(count)