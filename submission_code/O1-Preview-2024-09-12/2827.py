class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        from math import isqrt
        n = len(nums)
        if n == 1:
            return True
        if any(num == 1 for num in nums):
            return False
        
        max_num = max(nums)
        spf = [0] * (max_num + 1)
        for i in range(2, max_num + 1):
            if spf[i] == 0:
                for j in range(i, max_num + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

        parent = [i for i in range(n)]
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv

        factor_map = {}
        for idx, num in enumerate(nums):
            factors = set()
            x = num
            while x != 1:
                p = spf[x]
                factors.add(p)
                while x % p == 0:
                    x //= p
            for p in factors:
                if p in factor_map:
                    union(idx, factor_map[p])
                else:
                    factor_map[p] = idx

        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        return True