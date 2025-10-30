def count_no_ddos_strings(S):
    MOD = 998244353
    n = len(S)
    
    # Calculate total number of possible strings
    q = S.count('?')
    total_strings = pow(52, q, MOD)
    
    # Initialize DP arrays for each state
    # dp0: No pattern matched
    # dp1[k]: First uppercase letter k matched
    # dp2[k]: Two identical uppercase letters k matched
    # dp3[k]: Two identical uppercase letters k followed by lowercase matched
    # dp4: Complete DDoS-type subsequence matched
    dp0 = 1
    dp1 = [0] * 27
    dp2 = [0] * 27
    dp3 = [0] * 27
    dp4 = 0
    
    for i in range(n):
        char = S[i]
        new_dp0 = 0
        new_dp1 = [0] * 27
        new_dp2 = [0] * 27
        new_dp3 = [0] * 27
        new_dp4 = 0
        
        if char == '?':
            # '?' can be any of the 52 letters (26 uppercase, 26 lowercase)
            for val in range(52):
                is_upper = val < 26
                upper_idx = val + 1 if is_upper else 0
                
                # State 0: No pattern matched
                new_dp0 = (new_dp0 + dp0) % MOD
                
                # State 0 -> State 1: Match first uppercase letter
                if is_upper:
                    new_dp1[upper_idx] = (new_dp1[upper_idx] + dp0) % MOD
                
                # State 1: Process possible transitions
                for k in range(1, 27):
                    new_dp1[k] = (new_dp1[k] + dp1[k]) % MOD
                    
                    # State 1 -> State 2: Match second identical uppercase letter
                    if is_upper and k == upper_idx:
                        new_dp2[k] = (new_dp2[k] + dp1[k]) % MOD
                
                # State 2: Process possible transitions
                for k in range(1, 27):
                    new_dp2[k] = (new_dp2[k] + dp2[k]) % MOD
                    
                    # State 2 -> State 3: Match lowercase letter after two identical uppercase
                    if not is_upper:
                        new_dp3[k] = (new_dp3[k] + dp2[k]) % MOD
                
                # State 3: Process possible transitions
                for k in range(1, 27):
                    new_dp3[k] = (new_dp3[k] + dp3[k]) % MOD
                    
                    # State 3 -> State 4: Match final uppercase letter
                    if is_upper:
                        new_dp4 = (new_dp4 + dp3[k]) % MOD
                
                # State 4: Always stay in State 4
                new_dp4 = (new_dp4 + dp4) % MOD
        
        elif 'A' <= char <= 'Z':
            # Current character is uppercase
            upper_idx = ord(char) - ord('A') + 1
            
            # State 0
            new_dp0 = (new_dp0 + dp0) % MOD
            
            # State 0 -> State 1
            new_dp1[upper_idx] = (new_dp1[upper_idx] + dp0) % MOD
            
            # State 1
            for k in range(1, 27):
                new_dp1[k] = (new_dp1[k] + dp1[k]) % MOD
                
                # State 1 -> State 2
                if k == upper_idx:
                    new_dp2[k] = (new_dp2[k] + dp1[k]) % MOD
            
            # State 2
            for k in range(1, 27):
                new_dp2[k] = (new_dp2[k] + dp2[k]) % MOD
            
            # State 3
            for k in range(1, 27):
                new_dp3[k] = (new_dp3[k] + dp3[k]) % MOD
                
                # State 3 -> State 4
                new_dp4 = (new_dp4 + dp3[k]) % MOD
            
            # State 4
            new_dp4 = (new_dp4 + dp4) % MOD
        
        else:  # lowercase
            # State 0
            new_dp0 = (new_dp0 + dp0) % MOD
            
            # State 1
            for k in range(1, 27):
                new_dp1[k] = (new_dp1[k] + dp1[k]) % MOD
            
            # State 2
            for k in range(1, 27):
                new_dp2[k] = (new_dp2[k] + dp2[k]) % MOD
                
                # State 2 -> State 3
                new_dp3[k] = (new_dp3[k] + dp2[k]) % MOD
            
            # State 3
            for k in range(1, 27):
                new_dp3[k] = (new_dp3[k] + dp3[k]) % MOD
            
            # State 4
            new_dp4 = (new_dp4 + dp4) % MOD
        
        dp0 = new_dp0
        dp1 = new_dp1
        dp2 = new_dp2
        dp3 = new_dp3
        dp4 = new_dp4
    
    # Count strings with a DDoS-type subsequence
    strings_with_ddos = dp4
    
    # Calculate strings without a DDoS-type subsequence
    result = (total_strings - strings_with_ddos) % MOD
    if result < 0:
        result += MOD
    return result

# Read input
S = input().strip()
result = count_no_ddos_strings(S)
print(result)