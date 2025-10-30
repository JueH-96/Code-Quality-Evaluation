class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        
        # If there's a 1 in the array, it can't connect to anything
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
        
        # Get prime factors of a number
        def get_prime_factors(num):
            factors = set()
            d = 2
            while d * d <= num:
                while num % d == 0:
                    factors.add(d)
                    num //= d
                d += 1
            if num > 1:
                factors.add(num)
            return factors
        
        # Map prime factors to indices that have them
        prime_to_indices = {}
        
        for i, num in enumerate(nums):
            prime_factors = get_prime_factors(num)
            for prime in prime_factors:
                if prime not in prime_to_indices:
                    prime_to_indices[prime] = []
                prime_to_indices[prime].append(i)
        
        # Union indices that share prime factors
        for indices in prime_to_indices.values():
            for i in range(1, len(indices)):
                union(indices[0], indices[i])
        
        # Check if all indices are in the same connected component
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        
        return True