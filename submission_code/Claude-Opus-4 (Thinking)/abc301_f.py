def solve():
    S = input().strip()
    MOD = 998244353
    n = len(S)
    
    from collections import defaultdict
    
    # State: tuple of 26 values (progress for each uppercase letter A-Z)
    dp = defaultdict(int)
    initial = tuple([0] * 26)
    dp[initial] = 1
    
    for i in range(n):
        new_dp = defaultdict(int)
        
        for state, cnt in dp.items():
            if S[i] == '?':
                # Try all 52 characters
                for c in range(52):
                    if c < 26:
                        ch = chr(ord('A') + c)
                        is_up = True
                    else:
                        ch = chr(ord('a') + c - 26)
                        is_up = False
                    
                    new_state = update_state(state, ch, is_up)
                    if new_state:
                        new_dp[new_state] = (new_dp[new_state] + cnt) % MOD
            else:
                ch = S[i]
                is_up = ch.isupper()
                new_state = update_state(state, ch, is_up)
                if new_state:
                    new_dp[new_state] = (new_dp[new_state] + cnt) % MOD
        
        dp = new_dp
    
    result = sum(dp.values()) % MOD
    print(result)

def update_state(state, char, is_upper):
    new_state = list(state)
    
    if is_upper:
        idx = ord(char) - ord('A')
        
        # Check if this would complete a DDoS pattern
        for i in range(26):
            if state[i] == 3:
                return None
        
        # Update states
        for i in range(26):
            if i == idx:
                if state[i] == 0:
                    new_state[i] = 1
                elif state[i] == 1:
                    new_state[i] = 2
                elif state[i] == 2:
                    new_state[i] = 2  # Stay at 2
            else:
                if state[i] == 1 or state[i] == 2:
                    new_state[i] = 0
    else:
        # Lowercase
        for i in range(26):
            if state[i] == 1:
                new_state[i] = 0
            elif state[i] == 2:
                new_state[i] = 3
            elif state[i] == 3:
                new_state[i] = 0
    
    return tuple(new_state)

solve()