def count_possible_xor_values(nums):
    n = len(nums)
    possible_xors = set()
    visited = set()
    
    def dfs(bagcounts):
        key = tuple(bagcounts)
        if key in visited:
            return
        
        visited.add(key)
        
        # Calculate XOR of current configuration
        xor_value = 0
        for i in range(n):
            xor_value ^= bagcounts[i]
        possible_xors.add(xor_value)
        
        # Try all possible operations
        for i in range(n):
            for j in range(n):
                if i != j and bagcounts[i] > 0:  # Can only move from non-empty bag
                    new_bagcounts = list(bagcounts)
                    new_bagcounts[j] += new_bagcounts[i]
                    new_bagcounts[i] = 0
                    dfs(tuple(new_bagcounts))
    
    dfs(tuple(nums))
    return len(possible_xors)

# Read input
n = int(input())
nums = list(map(int, input().split()))
print(count_possible_xor_values(nums))