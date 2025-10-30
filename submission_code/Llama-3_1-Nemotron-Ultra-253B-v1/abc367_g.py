import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    max_xor = 1 << 20
    pow_x = [pow(x, K, MOD) for x in range(max_xor)]
    
    dp = [defaultdict(int) for _ in range(M)]
    dp[0][0] = 1
    
    for a in A:
        temp = [defaultdict(int) for _ in range(M)]
        for l in range(M):
            for x, cnt in dp[l].items():
                new_l = (l + 1) % M
                new_x = x ^ a
                temp[new_l][new_x] = (temp[new_l][new_x] + cnt) % MOD
        for l in range(M):
            for x, cnt in temp[l].items():
                dp[l][x] = (dp[l][x] + cnt) % MOD
    
    total = 0
    for x, cnt in dp[0].items():
        total = (total + pow_x[x] * cnt) % MOD
    total = (total - pow(0, K, MOD)) % MOD
    print(total)

if __name__ == "__main__":
    main()