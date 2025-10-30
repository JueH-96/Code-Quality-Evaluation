mod = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    inv_n = pow(n, mod-2, mod)
    
    sufA = [0] * (n+2)
    sufA[n] = 0
    for i in range(n-1, -1, -1):
        sufA[i] = (sufA[i+1] + A[i]) % mod
        
    dp = [0] * (n+1)
    sufDP = [0] * (n+2)
    
    for i in range(n-1, -1, -1):
        term = (sufA[i] + sufDP[i+1]) % mod
        dp[i] = term * inv_n % mod
        sufDP[i] = (dp[i] + sufDP[i+1]) % mod
        
    print(dp[0])

if __name__ == "__main__":
    main()