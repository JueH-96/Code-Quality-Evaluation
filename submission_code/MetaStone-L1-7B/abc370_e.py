from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    MOD = 998244353

    # Compute prefix sums
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i-1] + A[i-1]

    dp = [0] * (N + 1)
    dp[0] = 1
    pre_sum = [0] * (N + 1)
    pre_sum[0] = 1
    sum_map = defaultdict(int)
    sum_map[prefix[0]] = dp[0]

    for i in range(1, N + 1):
        current = prefix[i]
        target = current - K
        subtract = sum_map.get(target, 0)
        dp_i = (pre_sum[i-1] - subtract) % MOD
        dp[i] = dp_i
        pre_sum[i] = (pre_sum[i-1] + dp_i) % MOD
        sum_map[current] = (sum_map.get(current, 0) + dp_i) % MOD

    print(dp[N] % MOD)

if __name__ == '__main__':
    main()