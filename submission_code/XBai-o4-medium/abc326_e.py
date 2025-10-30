import sys

MOD = 998244353

def main():
    data = list(map(int, sys.stdin.read().split()))
    N = data[0]
    A = data[1:N+1]
    
    inv_N = pow(N, MOD-2, MOD)
    total = 0
    
    for k in range(N, 0, -1):
        dp = (total * inv_N) % MOD
        total = (total + A[k-1] + dp) % MOD
    
    ans = (total * inv_N) % MOD
    print(ans)

if __name__ == '__main__':
    main()