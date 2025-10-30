mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 0:
        print(0)
        return
        
    dp = [0] * (1 << 11)
    dp[1] = 1  # Start with only sum 0 achievable (bit0 set)
    mask_all = (1 << 11) - 1
    
    for a in A:
        new_dp = [0] * (1 << 11)
        T = min(10, a)
        inv_a = pow(a, mod-2, mod)
        base = (a - T) * inv_a % mod
        
        for s in range(1 << 11):
            if dp[s] == 0:
                continue
            new_dp[s] = (new_dp[s] + dp[s] * base) % mod
            for x in range(1, T + 1):
                shifted = s << x
                ns = s | (shifted & mask_all)
                new_dp[ns] = (new_dp[ns] + dp[s] * inv_a) % mod
        dp = new_dp
        
    ans = 0
    for s in range(1 << 11):
        if s & (1 << 10):
            ans = (ans + dp[s]) % mod
    print(ans)

if __name__ == "__main__":
    main()