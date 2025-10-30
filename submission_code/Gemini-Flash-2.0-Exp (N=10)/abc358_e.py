def solve():
    k = int(input())
    c = list(map(int, input().split()))
    mod = 998244353

    ans = 0
    for length in range(1, k + 1):
        dp = [0] * (length + 1)
        dp[0] = 1

        for char_limit in c:
            new_dp = [0] * (length + 1)
            for i in range(length + 1):
                for j in range(min(i, char_limit) + 1):
                    new_dp[i] = (new_dp[i] + dp[i-j]) % mod
            dp = new_dp
        ans = (ans + dp[length]) % mod
    print(ans)

solve()