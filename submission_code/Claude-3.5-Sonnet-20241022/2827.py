class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        if 1 in nums:
            return False
            
        # Find prime factors for each number
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
        
        # Union find implementation
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
        
        # Map prime factors to indices
        prime_to_index = {}
        for i, num in enumerate(nums):
            factors = get_prime_factors(num)
            for prime in factors:
                if prime in prime_to_index:
                    union(i, prime_to_index[prime])
                else:
                    prime_to_index[prime] = i
        
        # Check if all indices are connected
        root = find(0)
        return all(find(i) == root for i in range(n))