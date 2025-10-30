# YOUR CODE HERE
n, k = map(int, input().split())
r = list(map(int, input().split()))

ans = []
def dfs(idx, cur):
    if idx == n:
        if sum(cur) % k == 0:
            ans.append(cur.copy())
        return
    for i in range(1, r[idx] + 1):
        cur.append(i)
        dfs(idx + 1, cur)
        cur.pop()

dfs(0, [])

for a in ans:
    print(*a)