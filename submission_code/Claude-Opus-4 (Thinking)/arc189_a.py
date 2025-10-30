def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    MOD = 998244353
    
    # Initial configuration: cell i (1-indexed) has value i % 2
    # In 0-indexed: cell i has value (i + 1) % 2
    initial = [(i + 1) % 2 for i in range(n)]
    
    # Memoization using dictionary
    memo = {}
    
    def count_sequences(config):
        # Convert to tuple for hashing
        key = tuple(config)
        if key in memo:
            return memo[key]
        
        # Base case: reached target configuration
        if list(config) == a:
            memo[key] = 1
            return 1
        
        total = 0
        
        # Try all possible operations
        for l in range(n):
            for r in range(l + 2, n):  # Need l+1 < r
                if config[l] == config[r]:
                    # Check if all values between l and r are different from config[l]
                    valid = True
                    for i in range(l + 1, r):
                        if config[i] == config[l]:
                            valid = False
                            break
                    
                    if valid:
                        # Apply operation: set all values between l and r to config[l]
                        new_config = list(config)
                        for i in range(l + 1, r):
                            new_config[i] = config[l]
                        
                        # Recursively count sequences from new configuration
                        total = (total + count_sequences(new_config)) % MOD
        
        memo[key] = total
        return total
    
    result = count_sequences(tuple(initial))
    print(result)

solve()