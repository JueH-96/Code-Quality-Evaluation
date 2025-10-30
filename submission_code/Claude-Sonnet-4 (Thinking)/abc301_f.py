s = input().strip()
n = len(s)
MOD = 998244353

# dp[state][letter] = number of ways to be in this state
# state 0: no progress (letter irrelevant, use index 26)  
# state 1: found one uppercase letter
# state 2: found two identical uppercase letters
# state 3: found complete prefix of DDoS (letter irrelevant, use index 26)

dp = [[0 for _ in range(27)] for _ in range(4)]
dp[0][26] = 1  # Initially in state 0

for char in s:
    new_dp = [[0 for _ in range(27)] for _ in range(4)]
    
    if char == '?':
        # Try all 52 possible replacements
        for c in range(52):
            if c < 26:  # uppercase letter
                # From state 0 -> state 1[c]
                new_dp[1][c] = (new_dp[1][c] + dp[0][26]) % MOD
                
                # From state 1[d] -> state 2[c] if d==c, else state 1[c]
                for d in range(26):
                    if d == c:
                        new_dp[2][c] = (new_dp[2][c] + dp[1][d]) % MOD
                    else:
                        new_dp[1][c] = (new_dp[1][c] + dp[1][d]) % MOD
                
                # From state 2[d] -> state 1[c] 
                for d in range(26):
                    new_dp[1][c] = (new_dp[1][c] + dp[2][d]) % MOD
                
                # From state 3: uppercase completes DDoS (invalid)
                
            else:  # lowercase letter
                # From state 0 -> state 0
                new_dp[0][26] = (new_dp[0][26] + dp[0][26]) % MOD
                
                # From state 1[d] -> state 0
                for d in range(26):
                    new_dp[0][26] = (new_dp[0][26] + dp[1][d]) % MOD
                
                # From state 2[d] -> state 3
                for d in range(26):
                    new_dp[3][26] = (new_dp[3][26] + dp[2][d]) % MOD
                
                # From state 3 -> state 3
                new_dp[3][26] = (new_dp[3][26] + dp[3][26]) % MOD
    else:
        # Fixed character
        if char.isupper():
            c = ord(char) - ord('A')
            
            # From state 0 -> state 1[c]
            new_dp[1][c] = (new_dp[1][c] + dp[0][26]) % MOD
            
            # From state 1[d] -> state 2[c] if d==c, else state 1[c]
            for d in range(26):
                if d == c:
                    new_dp[2][c] = (new_dp[2][c] + dp[1][d]) % MOD
                else:
                    new_dp[1][c] = (new_dp[1][c] + dp[1][d]) % MOD
            
            # From state 2[d] -> state 1[c]
            for d in range(26):
                new_dp[1][c] = (new_dp[1][c] + dp[2][d]) % MOD
                
        else:  # lowercase letter
            # From state 0 -> state 0
            new_dp[0][26] = (new_dp[0][26] + dp[0][26]) % MOD
            
            # From state 1[d] -> state 0
            for d in range(26):
                new_dp[0][26] = (new_dp[0][26] + dp[1][d]) % MOD
            
            # From state 2[d] -> state 3
            for d in range(26):
                new_dp[3][26] = (new_dp[3][26] + dp[2][d]) % MOD
            
            # From state 3 -> state 3
            new_dp[3][26] = (new_dp[3][26] + dp[3][26]) % MOD
    
    dp = new_dp

# Sum valid states (0, 1, 2 but not 3)
result = dp[0][26]
for c in range(26):
    result = (result + dp[1][c] + dp[2][c]) % MOD

print(result)