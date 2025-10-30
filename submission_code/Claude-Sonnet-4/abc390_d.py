def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    memo = {}
    
    def get_possible_xors(mask):
        if mask in memo:
            return memo[mask]
        
        if mask == 0:
            return {0}
        
        result = set()
        
        # Try all possible ways to choose the first group
        submask = mask
        while submask > 0:
            # Calculate sum of this group
            group_sum = 0
            for i in range(n):
                if submask & (1 << i):
                    group_sum += a[i]
            
            # Get XOR values for remaining elements
            remaining = mask ^ submask
            remaining_xors = get_possible_xors(remaining)
            
            # Combine with current group
            for xor_val in remaining_xors:
                result.add(group_sum ^ xor_val)
            
            submask = (submask - 1) & mask
        
        memo[mask] = result
        return result
    
    all_mask = (1 << n) - 1
    possible_xors = get_possible_xors(all_mask)
    return len(possible_xors)

print(solve())