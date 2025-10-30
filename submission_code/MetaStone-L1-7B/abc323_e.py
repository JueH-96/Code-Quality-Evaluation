import math

def main():
    mod = 998244353
    N, X = map(int, input().split())
    T = list(map(int, input().split()))
    T1 = T[0]
    
    # Calculate the lower bound
    lower = math.ceil((X + 0.5) - T1)
    lower = max(0, lower)
    upper = X
    
    max_s = X
    inv_N = pow(N, mod - 2, mod)
    
    # Initialize dp array
    dp = [0] * (max_s + 1)
    dp[0] = 1
    
    for s in range(max_s + 1):
        if dp[s] == 0:
            continue
        for ti in T:
            next_s = s + ti
            if next_s > max_s:
                continue
            dp[next_s] = (dp[next_s] + dp[s] * inv_N) % mod
    
    # Calculate the result
    res = 0
    for s in range(lower, upper + 1):
        res = (res + dp[s]) % mod
    
    res = res * inv_N % mod
    print(res)

if __name__ == '__main__':
    main()