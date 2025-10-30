def solve():
    N = int(input())
    parents = list(map(int, input().split()))
    a = list(map(int, input().split()))

    adj = [[] for _ in range(N + 1)]
    for i in range(N):
        adj[parents[i]].append(i + 1)

    total_a = sum(a)
    mod = 998244353

    def power(a, b):
        res = 1
        a %= mod
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % mod
            a = (a * a) % mod
            b //= 2
        return res

    def inverse(n):
        return power(n, mod - 2)

    subtree_sum = [0] * (N + 1)

    def dfs(u):
        current_sum = 0
        if 1 <= u <= N:
            current_sum = a[u - 1]
        for v in adj[u]:
            current_sum = (current_sum + dfs(v)) % mod
        subtree_sum[u] = current_sum
        return current_sum

    dfs(0)

    expected_ops = 0
    inv_total_a = inverse(total_a)

    for i in range(1, N + 1):
        prob_treasure_in_subtree = (subtree_sum[i] * inv_total_a) % mod
        expected_ops = (expected_ops + prob_treasure_in_subtree) % mod

    print(expected_ops)

T = int(input())
for _ in range(T):
    solve()