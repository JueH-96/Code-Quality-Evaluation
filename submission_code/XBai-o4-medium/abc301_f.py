MOD = 998244353

def main():
    import sys
    S = sys.stdin.readline().strip()
    dp0 = 1
    dp1 = [0] * 26
    dp2 = 0
    dp3 = 0
    
    for c in S:
        if c == '?':
            upper_x = [1] * 26
            upper_total = 26
            lower_total = 26
        elif c.isupper():
            idx = ord(c) - ord('A')
            upper_x = [0] * 26
            upper_x[idx] = 1
            upper_total = 1
            lower_total = 0
        else:
            upper_x = [0] * 26
            upper_total = 0
            lower_total = 1
        
        new_dp0 = (dp0 * lower_total) % MOD
        new_dp1 = [ (dp0 * upper_x[x]) % MOD for x in range(26) ]
        new_dp2 = 0
        new_dp3 = 0
        
        for x in range(26):
            current = dp1[x]
            if current == 0:
                continue
            new_dp2 += current * upper_x[x]
            term = current * ((upper_total - upper_x[x]) + lower_total)
            new_dp1[x] = (new_dp1[x] + term) % MOD
        
        new_dp2 %= MOD
        new_dp2 += (dp2 * upper_total) % MOD
        new_dp2 %= MOD
        new_dp3 += (dp2 * lower_total) % MOD
        new_dp3 %= MOD
        new_dp3 += (dp3 * lower_total) % MOD
        new_dp3 %= MOD
        
        dp0, dp1, dp2, dp3 = new_dp0, new_dp1, new_dp2, new_dp3
    
    ans = (dp0 + sum(dp1) + dp2 + dp3) % MOD
    print(ans)

if __name__ == "__main__":
    main()