# YOUR CODE HERE
def solve():
    n, m, c, k = map(int, input().split())
    a = list(map(int, input().split()))

    ans = 0
    for kk in range(k):
        min_val = float('inf')
        for i in range(n):
            val = (c * kk + a[i]) % m
            min_val = min(min_val, val)
        ans += min_val
    print(ans)

solve()