from typing import List

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        parent = list(range(len(nums)))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv
        
        sorted_indices = sorted(range(len(nums)), key=lambda x: nums[x])
        
        for i in range(1, len(sorted_indices)):
            if nums[sorted_indices[i]] - nums[sorted_indices[i-1]] <= limit:
                union(sorted_indices[i], sorted_indices[i-1])
        
        groups = {}
        for idx in range(len(nums)):
            root = find(idx)
            if root not in groups:
                groups[root] = []
            groups[root].append(idx)
        
        result = nums.copy()
        for group in groups.values():
            values = sorted(nums[i] for i in group)
            indices = sorted(group)
            for i, val in zip(indices, values):
                result[i] = val
        
        return result