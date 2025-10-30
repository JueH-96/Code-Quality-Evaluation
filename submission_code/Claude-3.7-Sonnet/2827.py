class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        
        # Edge cases
        if n == 1:
            return True
        if 1 in nums:
            return False
        
        # Union-Find data structure
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Map each prime factor to the first index where it appears
        factor_to_index = {}
        
        for i in range(n):
            num = nums[i]
            
            # Find all prime factors of num
            j = 2
            while j * j <= num:
                if num % j == 0:
                    # j is a prime factor
                    if j in factor_to_index:
                        union(i, factor_to_index[j])
                    else:
                        factor_to_index[j] = i
                    
                    # Remove all occurrences of this prime factor
                    while num % j == 0:
                        num //= j
                
                j += 1
            
            # If num is still greater than 1, it's a prime
            if num > 1:
                if num in factor_to_index:
                    union(i, factor_to_index[num])
                else:
                    factor_to_index[num] = i
        
        # Check if all indices are in the same connected component
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        
        return True