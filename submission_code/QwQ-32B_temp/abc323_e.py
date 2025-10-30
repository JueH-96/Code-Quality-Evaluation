mod = 998244353

def main():
    import sys
    N, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))
    T1 = T[0]
    
    S_min = (X - T1) + 1
    S_min = max(S_min, 0)
    S_max = X
    
    if S_min > S_max:
        print(0)
        return
    
    inv_N = pow(N, mod-2, mod)
    DP = [0] * (X + 1)
    DP[0] = 1
    
    for s in range(X + 1):
        if DP[s] == 0:
            continue
        for t in T:
            next_s = s + t
            if next_s > X:
                continue
            DP[next_s] = (DP[next_s] + DP[s] * inv_N) % mod
    
    total = 0
    for s in range(S_min, S_max + 1):
        total = (total + DP[s]) % mod
    
    ans = total * inv_N % mod
    print(ans)

if __name__ == "__main__":
    main()