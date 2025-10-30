class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        
        # Check if there's a 1 in the array
        if 1 in nums:
            return False
        
        # Union-Find data structure
        parent = list(range(len(nums)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Get prime factors for a number
        def get_prime_factors(n):
            factors = []
            d = 2
            while d * d <= n:
                if n % d == 0:
                    factors.append(d)
                    while n % d == 0:
                        n //= d
                d += 1
            if n > 1:
                factors.append(n)
            return factors
        
        # Map prime factor to a representative index
        prime_to_index = {}
        
        for i, num in enumerate(nums):
            prime_factors = get_prime_factors(num)
            for prime in prime_factors:
                if prime in prime_to_index:
                    # Connect this index with the representative index for this prime
                    union(i, prime_to_index[prime])
                else:
                    prime_to_index[prime] = i
        
        # Check if all indices are connected
        root = find(0)
        for i in range(1, len(nums)):
            if find(i) != root:
                return False
        
        return True