# YOUR CODE HERE
MOD = 998244353

def solve():
    N, Q = map(int, input().split())
    operations = [tuple(map(int, input().split())) for _ in range(Q)]

    dp = [1] * (N + 1)
    max_val = [0] * (N + 1)

    for P, V in operations:
        left_count = sum(dp[1:P+1]) % MOD
        right_count = sum(dp[P:]) % MOD

        new_dp = [0] * (N + 1)
        new_max_val = [0] * (N + 1)

        for i in range(1, N + 1):
            if i <= P and max_val[i] <= V:
                new_dp[i] = (new_dp[i] + left_count) % MOD
                new_max_val[i] = V
            if i >= P and max_val[i] <= V:
                new_dp[i] = (new_dp[i] + right_count) % MOD
                new_max_val[i] = V

        dp = new_dp
        max_val = new_max_val

    print(sum(dp) % MOD)

solve()