N = int(input())
A = list(map(int, input().split()))

ans = 0
for i in range(N):
    if i == 0:
        ans += 1
        continue
    d = A[i] - A[i-1]
    cnt = 1
    if d == 0:
        ans += 2
        continue
    for j in range(i-1, -1, -1):
        if A[j] - A[j+1] == d:
            cnt += 1
        else:
            break
    ans += cnt

print(ans)