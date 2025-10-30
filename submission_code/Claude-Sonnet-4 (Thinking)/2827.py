class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        
        if 1 in nums:
            return False
        
        def get_prime_factors(n):
            factors = set()
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors.add(d)
                    n //= d
                d += 1
            if n > 1:
                factors.add(n)
            return factors
        
        # Union-Find
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Connect each index to its prime factors
        for i, num in enumerate(nums):
            prime_factors = get_prime_factors(num)
            for factor in prime_factors:
                union(i, f"factor_{factor}")
        
        # Check if all indices are in the same component
        root = find(0)
        for i in range(1, len(nums)):
            if find(i) != root:
                return False
        
        return True