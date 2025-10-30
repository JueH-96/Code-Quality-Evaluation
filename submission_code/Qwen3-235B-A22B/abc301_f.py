MOD = 998244353

def main():
    import sys
    S = sys.stdin.readline().strip()
    
    dp0 = 1
    dp1 = [0] * 26
    dp2 = [0] * 26
    dp3 = [0] * 26
    
    for char in S:
        if char == '?':
            up_count = [1] * 26
            low_total = 26
        else:
            if 'A' <= char <= 'Z':
                idx = ord(char) - ord('A')
                up_count = [0] * 26
                up_count[idx] = 1
                low_total = 0
            elif 'a' <= char <= 'z':
                up_count = [0] * 26
                low_total = 1
            else:
                up_count = [0] * 26
                low_total = 0
        
        up_total = sum(up_count)
        len_X = up_total + low_total
        
        new_dp0 = (dp0 * len_X) % MOD
        
        new_dp1 = [0] * 26
        for c in range(26):
            term1 = (dp0 * up_count[c]) % MOD
            term2 = (dp1[c] * len_X) % MOD
            new_dp1[c] = (term1 + term2) % MOD
        
        new_dp2 = [0] * 26
        for c in range(26):
            term1 = (dp1[c] * up_count[c]) % MOD
            term2 = (dp2[c] * len_X) % MOD
            new_dp2[c] = (term1 + term2) % MOD
        
        new_dp3 = [0] * 26
        for c in range(26):
            term1 = (dp2[c] * low_total) % MOD
            term2 = (dp3[c] * len_X) % MOD
            new_dp3[c] = (term1 + term2) % MOD
        
        dp0, dp1, dp2, dp3 = new_dp0, new_dp1, new_dp2, new_dp3
    
    total = dp0
    for c in range(26):
        total = (total + dp1[c]) % MOD
        total = (total + dp2[c]) % MOD
        total = (total + dp3[c]) % MOD
    
    print(total)

if __name__ == "__main__":
    main()