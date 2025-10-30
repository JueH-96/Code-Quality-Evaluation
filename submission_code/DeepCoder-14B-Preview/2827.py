class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 0:
            return False
        
        # Check for the presence of 1, which cannot form GCD > 1 with any other number
        if any(num == 1 for num in nums):
            return len(nums) == 1
        
        n = len(nums)
        parent = list(range(n))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u
        
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root == v_root:
                return
            parent[v_root] = u_root
        
        def get_unique_primes(x):
            factors = set()
            if x % 2 == 0:
                factors.add(2)
                while x % 2 == 0:
                    x = x // 2
            i = 3
            while i * i <= x:
                while x % i == 0:
                    factors.add(i)
                    x = x // i
                i += 2
            if x > 2:
                factors.add(x)
            return factors
        
        primes_map = {}
        for i in range(n):
            num = nums[i]
            factors = get_unique_primes(num)
            for p in factors:
                # Update the list of indices for this prime
                primes_map[p] = primes_map.get(p, []) + [i]
                # If there are at least two indices, connect the new one with the first
                if len(primes_map[p]) >= 2:
                    first_i = primes_map[p][0]
                    union(i, first_i)
        
        # Check if all indices are connected
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        
        return True