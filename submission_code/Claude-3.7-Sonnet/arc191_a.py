def max_after_operations(s, t):
    N = len(s)
    M = len(t)
    
    # We'll use memoization to avoid recalculating the same states
    memo = {}
    
    def dfs(idx, curr_s):
        """
        Recursive function to find the maximum possible value after performing operations
        idx: current operation index (0-indexed)
        curr_s: current state of string S
        """
        if idx == M:
            return int(curr_s)
        
        # Check if we've already calculated this state
        if (idx, curr_s) in memo:
            return memo[(idx, curr_s)]
        
        max_val = 0
        for i in range(N):
            # Try replacing position i with the current character from T
            new_s = curr_s[:i] + t[idx] + curr_s[i+1:]
            val = dfs(idx + 1, new_s)
            max_val = max(max_val, val)
        
        # Memoize the result
        memo[(idx, curr_s)] = max_val
        return max_val
    
    return dfs(0, s)

# Read input
N, M = map(int, input().split())
S = input().strip()
T = input().strip()

# Solve and print the result
result = max_after_operations(S, T)
print(result)