import sys
MOD = 998244353

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    K = int(input[idx]); idx += 1
    A = list(map(int, input[idx:idx+N]))
    
    SIZE = 1 << 20
    dp = [ [0]*SIZE for _ in range(M) ]
    dp[0][0] = 1  # empty subset

    for a in A:
        # Iterate in reverse to prevent overlapping updates
        for r in range(M-1, -1, -1):
            prev_r = (r - 1 + M) % M
            src = dp[prev_r]
            dst = dp[r]
            for x in range(SIZE):
                new_x = x ^ a
                dst[new_x] = (dst[new_x] + src[x]) % MOD
    
    res = 0
    for x in range(SIZE):
        xk = pow(x, K, MOD)
        res = (res + xk * dp[0][x]) % MOD
    print(res)

if __name__ == "__main__":
    main()