def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    mod = 998244353
    max_xor = 1 << 20

    dp = [[0] * max_xor for _ in range(m)]
    dp[0][0] = 1

    for val in a:
        new_dp = [[0] * max_xor for _ in range(m)]
        for length_mod in range(m):
            for xor_sum in range(max_xor):
                if dp[length_mod][xor_sum] > 0:
                    new_length_mod = (length_mod + 1) % m
                    new_xor_sum = xor_sum ^ val
                    new_dp[new_length_mod][new_xor_sum] = (new_dp[new_length_mod][new_xor_sum] + dp[length_mod][xor_sum]) % mod
                    new_dp[length_mod][xor_sum] = (new_dp[length_mod][xor_sum] + dp[length_mod][xor_sum]) % mod
        dp = new_dp

    total_score = 0
    for xor_val in range(1, max_xor):
        count = dp[0][xor_val]
        if count > 0:
            score = pow(xor_val, k, mod)
            total_score = (total_score + count * score) % mod

    return total_score

result = solve()
print(result)