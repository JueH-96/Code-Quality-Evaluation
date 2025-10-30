# YOUR CODE HERE
n, m, k = map(int, input().split())
a = list(map(int, input().split()))

ans = []
for i in range(n):
    ok = False
    for x in range(k + 1):
        b = list(a)
        b[i] += x
        cnt = 0
        for j in range(n):
            if b[j] > b[i]:
                cnt += 1
        if cnt < m:
            ok = True
            ans.append(x)
            break
    if not ok:
        ans.append(-1)

remaining_votes = k - sum(a)
for i in range(n):
    if ans[i] != -1 and ans[i] > remaining_votes:
        ans[i] = -1

print(*ans)