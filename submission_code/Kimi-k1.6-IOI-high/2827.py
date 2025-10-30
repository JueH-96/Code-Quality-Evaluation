class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        def get_prime_factors(n):
            factors = set()
            if n % 2 == 0:
                factors.add(2)
                while n % 2 == 0:
                    n //= 2
            i = 3
            while i * i <= n:
                if n % i == 0:
                    factors.add(i)
                    while n % i == 0:
                        n //= i
                i += 2
            if n > 1:
                factors.add(n)
            return factors
        
        n = len(nums)
        if n == 1:
            return True
        if 1 in nums:
            return False
        
        primes = set()
        for num in nums:
            factors = get_prime_factors(num)
            primes.update(factors)
        primes = list(primes)
        prime_to_id = {p: n + i for i, p in enumerate(primes)}
        uf_size = n + len(primes)
        
        parent = list(range(uf_size))
        rank = [0] * uf_size
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            x_root = find(x)
            y_root = find(y)
            if x_root == y_root:
                return
            if rank[x_root] < rank[y_root]:
                parent[x_root] = y_root
            else:
                parent[y_root] = x_root
                if rank[x_root] == rank[y_root]:
                    rank[x_root] += 1
        
        for i in range(n):
            num = nums[i]
            factors = get_prime_factors(num)
            for p in factors:
                p_id = prime_to_id[p]
                union(i, p_id)
        
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        return True