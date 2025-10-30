# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)

    ans = 0
    for i in range(n):
        for j in adj[i]:
            if i < j:
                for k in adj[j]:
                    if j < k:
                        if k not in adj[i]:
                            ans += 1

    print(ans * 2)

solve()