import sys

def solve():
    S = sys.stdin.readline().strip()
    N = len(S)
    MOD = 998244353

    current_dp0 = 1
    current_dp1 = [0] * 26
    current_dp2 = [0] * 26
    current_dp3 = [0] * 26

    for i in range(N):
        s_char = S[i]
        
        next_dp0 = 0
        next_dp1 = [0] * 26
        next_dp2 = [0] * 26
        next_dp3 = [0] * 26

        if s_char == '?':
            # Case 1: S[i] is a lowercase letter (26 options)
            if current_dp0 > 0:
                next_dp0 = (next_dp0 + current_dp0 * 26) % MOD
            for X_code in range(26):
                if current_dp1[X_code] > 0:
                    next_dp1[X_code] = (next_dp1[X_code] + current_dp1[X_code] * 26) % MOD
                if current_dp2[X_code] > 0:
                    next_dp3[X_code] = (next_dp3[X_code] + current_dp2[X_code] * 26) % MOD
                if current_dp3[X_code] > 0:
                    next_dp3[X_code] = (next_dp3[X_code] + current_dp3[X_code] * 26) % MOD
            
            # Case 2: S[i] is an uppercase letter (26 options U_C)
            if current_dp0 > 0:
                for C_code in range(26): # S[i] becomes U_C
                    next_dp1[C_code] = (next_dp1[C_code] + current_dp0) % MOD
            for X_code in range(26):
                if current_dp1[X_code] > 0:
                    # S[i] becomes U_X (1 option)
                    next_dp2[X_code] = (next_dp2[X_code] + current_dp1[X_code]) % MOD
                    # S[i] becomes U_C where C != X (25 options)
                    next_dp1[X_code] = (next_dp1[X_code] + current_dp1[X_code] * 25) % MOD
                if current_dp2[X_code] > 0:
                    # S[i] becomes U_C (26 options)
                    next_dp2[X_code] = (next_dp2[X_code] + current_dp2[X_code] * 26) % MOD
            # From current_dp3: if S[i] becomes uppercase, it's a BAD path.
            
        else: # s_char is a fixed character
            char_val = s_char
            is_lower = 'a' <= char_val <= 'z'
            
            if current_dp0 > 0:
                if is_lower:
                    next_dp0 = (next_dp0 + current_dp0) % MOD
                else: # is_upper
                    char_U_code = ord(char_val) - ord('A')
                    next_dp1[char_U_code] = (next_dp1[char_U_code] + current_dp0) % MOD
            
            for X_code in range(26):
                if current_dp1[X_code] > 0:
                    if is_lower:
                        next_dp1[X_code] = (next_dp1[X_code] + current_dp1[X_code]) % MOD
                    else: # is_upper
                        current_char_U_code = ord(char_val) - ord('A')
                        if current_char_U_code == X_code:
                            next_dp2[X_code] = (next_dp2[X_code] + current_dp1[X_code]) % MOD
                        else:
                            next_dp1[X_code] = (next_dp1[X_code] + current_dp1[X_code]) % MOD
                
                if current_dp2[X_code] > 0:
                    if is_lower:
                        next_dp3[X_code] = (next_dp3[X_code] + current_dp2[X_code]) % MOD
                    else: # is_upper
                        next_dp2[X_code] = (next_dp2[X_code] + current_dp2[X_code]) % MOD

                if current_dp3[X_code] > 0:
                    if is_lower:
                        next_dp3[X_code] = (next_dp3[X_code] + current_dp3[X_code]) % MOD
                    # else is_upper: BAD path, do nothing
        
        current_dp0 = next_dp0
        current_dp1 = next_dp1
        current_dp2 = next_dp2
        current_dp3 = next_dp3

    ans = current_dp0
    for X_code in range(26):
        ans = (ans + current_dp1[X_code]) % MOD
        ans = (ans + current_dp2[X_code]) % MOD
        ans = (ans + current_dp3[X_code]) % MOD
        
    print(ans)

solve()