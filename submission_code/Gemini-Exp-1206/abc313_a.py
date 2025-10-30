N = int(input())
P = list(map(int, input().split()))
max_other = 0
for i in range(1, N):
    if P[i] > max_other:
        max_other = P[i]
ans = max(0, max_other - P[0] + 1)
print(ans)