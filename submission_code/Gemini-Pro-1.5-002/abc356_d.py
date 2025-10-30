# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    mod = 998244353
    ans = 0
    for i in range(60):
        if (m >> i) & 1:
            count = (n >> (i + 1)) * (1 << i)
            if (n >> i) & 1:
                count += (n & ((1 << i) - 1)) + 1
            ans += count
            ans %= mod
    print(ans)

solve()