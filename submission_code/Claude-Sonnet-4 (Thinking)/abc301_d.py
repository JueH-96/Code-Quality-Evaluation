def solve(S, N):
    # Check if the maximum possible value from S is ≤ N
    max_possible = int(S.replace('?', '1'), 2)
    if max_possible <= N:
        return max_possible
    
    # Check if any value from S is ≤ N
    min_possible = int(S.replace('?', '0'), 2)
    if min_possible > N:
        return -1
    
    # Handle the case where S is longer than the binary representation of N
    N_bin = bin(N)[2:]
    if len(S) > len(N_bin):
        # For the number to be ≤ N, leading bits must be 0
        if S[0] == '1':
            return -1  # Can't have leading 1 if S is longer than N_bin
        elif S[0] == '?':
            # Try with leading 0
            return solve('0' + S[1:], N)
        else:  # S[0] == '0'
            # Continue with the rest
            return solve(S[1:], N)
    
    # General DP approach
    from functools import lru_cache
    
    # Pad S to match N_bin length
    if len(S) < len(N_bin):
        S = '0' * (len(N_bin) - len(S)) + S
    
    @lru_cache(None)
    def dp(pos, tight):
        if pos == len(S):
            return 0
        
        limit = 1
        if tight and pos < len(N_bin):
            limit = int(N_bin[pos])
        
        max_val = -1
        
        candidates = []
        if S[pos] == '0':
            candidates = [0]
        elif S[pos] == '1':
            candidates = [1]
        else:  # S[pos] == '?'
            candidates = [0, 1]
        
        for digit in candidates:
            if digit > limit:
                continue
            
            new_tight = tight and (digit == limit)
            sub_val = dp(pos + 1, new_tight)
            
            if sub_val != -1:
                current_val = digit * (2 ** (len(S) - pos - 1)) + sub_val
                max_val = max(max_val, current_val)
        
        return max_val
    
    result = dp(0, True)
    return result if result != -1 and result <= N else -1

# Read inputs
S = input().strip()
N = int(input().strip())

# Solve and print the result
print(solve(S, N))