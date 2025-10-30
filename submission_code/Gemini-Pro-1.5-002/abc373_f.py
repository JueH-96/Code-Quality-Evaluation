# YOUR CODE HERE
def solve():
    n, w = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))

    max_k = 0
    for wi, vi in items:
        max_k = max(max_k, (vi + wi -1 ) // (2 * wi))
    
    max_k = min(max_k, w)

    dp = [-float('inf')] * (w + 1)
    dp[0] = 0

    for wi, vi in items:
        new_dp = [-float('inf')] * (w + 1)
        for j in range(w + 1):
            for k in range(min(max_k + 1, (j // wi) + 1)):
                if j - k * wi >= 0:
                    new_dp[j] = max(new_dp[j], dp[j - k * wi] + k * vi - k * k)
        dp = new_dp
    
    print(max(dp))


solve()