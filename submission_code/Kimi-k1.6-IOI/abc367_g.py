from collections import defaultdict

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    max_x = 1 << 20
    pow_x_k = [0] * max_x
    for x in range(max_x):
        pow_x_k[x] = pow(x, K, MOD)
    
    dp = [defaultdict(int) for _ in range(M)]
    dp[0][0] = 1
    
    for a in A:
        for r in reversed(range(M)):
            items = list(dp[r].items())
            for x, cnt in items:
                new_r = (r + 1) % M
                new_x = x ^ a
                dp[new_r][new_x] = (dp[new_r][new_x] + cnt) % MOD
    
    ans = 0
    for x, cnt in dp[0].items():
        ans = (ans + cnt * pow_x_k[x]) % MOD
    print(ans)

if __name__ == "__main__":
    main()