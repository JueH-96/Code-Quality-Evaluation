# YOUR CODE HERE
MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    total = 1
    for a in A:
        total *= a

    dp = [0] * 11
    dp[0] = 1

    for a in A:
        new_dp = [0] * 11
        for i in range(11):
            for j in range(1, min(a, 11 - i) + 1):
                new_dp[i + j] += dp[i]
        dp = new_dp

    favorable = sum(dp)
    probability = favorable * mod_inverse(total) % MOD

    print(probability)

solve()