class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        
        # If any number is 1, we can't traverse from/to it
        if 1 in nums:
            return False
        
        # Union-Find data structure
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Map each prime to the first index that has it as a factor
        prime_to_index = {}
        
        # For each number, find its prime factors and union indices
        for i in range(n):
            num = nums[i]
            
            # Find prime factors of num
            # Check factor 2
            if num % 2 == 0:
                if 2 in prime_to_index:
                    union(i, prime_to_index[2])
                else:
                    prime_to_index[2] = i
                while num % 2 == 0:
                    num //= 2
            
            # Check odd factors
            factor = 3
            while factor * factor <= num:
                if num % factor == 0:
                    if factor in prime_to_index:
                        union(i, prime_to_index[factor])
                    else:
                        prime_to_index[factor] = i
                    while num % factor == 0:
                        num //= factor
                factor += 2
            
            # If num > 1, then it's a prime factor
            if num > 1:
                if num in prime_to_index:
                    union(i, prime_to_index[num])
                else:
                    prime_to_index[num] = i
        
        # Check if all indices are in the same connected component
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        
        return True