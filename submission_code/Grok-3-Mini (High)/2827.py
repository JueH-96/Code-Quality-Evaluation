from typing import List
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        if 1 in nums:
            return False
        
        # Precompute smallest prime factor (spf)
        max_num = 100001
        spf = [0] * max_num
        for i in range(2, max_num):
            if spf[i] == 0:
                for j in range(i, max_num, i):
                    if spf[j] == 0:
                        spf[j] = i
        
        # Function to get prime factors
        def get_prime_factors(num):
            factors = []
            while num > 1:
                p = spf[num]
                factors.append(p)
                while num % p == 0:
                    num //= p
            return factors
        
        # Build map from prime to indices
        prime_to_indices = defaultdict(list)
        for i in range(n):
            factors = get_prime_factors(nums[i])
            for prime in factors:
                prime_to_indices[prime].append(i)
        
        # Union-Find setup
        parent = [i for i in range(n)]
        rank = [0] * n
        
        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            # Path compression
            while x != root:
                next_node = parent[x]
                parent[x] = root
                x = next_node
            return root
        
        def union_op(x, y):  # Renamed to avoid keyword conflict
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                if rank[rootX] > rank[rootY]:
                    parent[rootY] = rootX
                elif rank[rootX] < rank[rootY]:
                    parent[rootX] = rootY
                else:
                    parent[rootY] = rootX
                    rank[rootX] += 1
        
        # Union indices for each prime
        for indices in prime_to_indices.values():
            if len(indices) >= 2:
                # Union all to the first one
                first = indices[0]
                for idx in indices[1:]:
                    union_op(first, idx)  # Use renamed union_op
        
        # Check if all nodes have the same root
        root0 = find(0)
        for i in range(1, n):
            if find(i) != root0:
                return False
        return True