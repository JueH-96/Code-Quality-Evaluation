from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        # If there's only one element, trivially return True
        if n == 1:
            return True
        
        # If there's any '1' in the array (and array length > 1), no edges will connect it to others
        # or to itself (gcd(1,1)=1 not > 1), so return False
        for x in nums:
            if x == 1:
                return False

        # Precompute smallest prime factors (spf) for all numbers up to 10^5
        max_num = 10**5
        spf = [0]*(max_num+1)  # spf[x] will store the smallest prime factor of x
        
        def build_spf():
            spf[1] = 1
            for i in range(2, max_num+1):
                if spf[i] == 0:  # i is prime
                    spf[i] = i
                    for j in range(i*i, max_num+1, i):
                        if spf[j] == 0:
                            spf[j] = i
        
        build_spf()
        
        # Union-Find (Disjoint Set) data structure
        parent = list(range(n))
        rank = [0]*n
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a: int, b: int) -> None:
            rootA = find(a)
            rootB = find(b)
            if rootA != rootB:
                if rank[rootA] < rank[rootB]:
                    rootA, rootB = rootB, rootA
                parent[rootB] = rootA
                if rank[rootA] == rank[rootB]:
                    rank[rootA] += 1
        
        # For each prime factor, keep track of the indices that contain it
        prime_map = {}
        
        # Factor each number and record the index in prime_map
        for i, val in enumerate(nums):
            # Factor val using spf
            factors_set = set()
            x = val
            while x > 1:
                factors_set.add(spf[x])
                x //= spf[x]
            # Map each prime factor to our current index
            for f in factors_set:
                if f not in prime_map:
                    prime_map[f] = []
                prime_map[f].append(i)
        
        # For each prime factor, union all indices that have it
        for f, indices in prime_map.items():
            if len(indices) > 1:
                base_idx = indices[0]
                for j in range(1, len(indices)):
                    union(base_idx, indices[j])
        
        # Finally, check if all indices are in the same connected component
        root0 = find(0)
        for i in range(1, n):
            if find(i) != root0:
                return False
        return True