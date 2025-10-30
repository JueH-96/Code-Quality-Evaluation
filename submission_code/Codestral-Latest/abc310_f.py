MOD = 998244353

def mod_inverse(a, m):
    return pow(a, m - 2, m)

def count_ways_to_sum_10(A):
    dp = [0] * 11
    dp[0] = 1
    for a in A:
        for j in range(10, 0, -1):
            for k in range(1, a + 1):
                if j - k >= 0:
                    dp[j] += dp[j - k]
                    dp[j] %= MOD
    return dp[10]

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:N+1]))

    total_ways = 1
    for a in A:
        total_ways = (total_ways * (a + 1)) % MOD

    ways_to_sum_10 = count_ways_to_sum_10(A)

    if total_ways == 0:
        print(0)
    else:
        print((ways_to_sum_10 * mod_inverse(total_ways, MOD)) % MOD)

if __name__ == "__main__":
    main()