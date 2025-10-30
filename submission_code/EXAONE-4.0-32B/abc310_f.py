mod = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    total = 1
    for a in A:
        total = (total * a) % mod
        
    factor_group2 = 1
    for a in A:
        if a > 10:
            factor_group2 = (factor_group2 * (a - 10)) % mod
            
    dp = [0] * (1 << 11)
    dp[1] = 1
    
    for a in A:
        m = min(a, 10)
        new_dp = [0] * (1 << 11)
        for state in range(1 << 11):
            if dp[state] == 0:
                continue
            for v in range(1, m + 1):
                new_state = state | (state << v)
                new_state &= (1 << 11) - 1
                new_dp[new_state] = (new_dp[new_state] + dp[state]) % mod
        dp = new_dp
        
    bad_group1 = 0
    target_bit = 1 << 10
    for state in range(1 << 11):
        if not (state & target_bit):
            bad_group1 = (bad_group1 + dp[state]) % mod
            
    bad_total = (factor_group2 * bad_group1) % mod
    favorable = (total - bad_total) % mod
    if favorable < 0:
        favorable += mod
        
    inv_total = pow(total, mod - 2, mod)
    ans = (favorable * inv_total) % mod
    print(ans)

if __name__ == "__main__":
    main()