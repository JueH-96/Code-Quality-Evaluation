class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Union-Find (Disjoint Set) implementation
        class UnionFind:
            def __init__(self, n: int):
                self.parent = list(range(n))
                self.rank = [0]*n
            
            def find(self, x: int) -> int:
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x: int, y: int) -> None:
                rx, ry = self.find(x), self.find(y)
                if rx != ry:
                    if self.rank[rx] < self.rank[ry]:
                        rx, ry = ry, rx
                    self.parent[ry] = rx
                    if self.rank[rx] == self.rank[ry]:
                        self.rank[rx] += 1
        
        # Step 1: Extract distinct values and sort them
        distinct_vals = sorted(set(nums))
        
        # Build a map from value -> index in distinct_vals
        val_to_index = {v: i for i, v in enumerate(distinct_vals)}
        
        # Step 2: Union consecutive values if their difference <= limit
        uf = UnionFind(len(distinct_vals))
        for i in range(len(distinct_vals) - 1):
            if distinct_vals[i+1] - distinct_vals[i] <= limit:
                uf.union(i, i+1)
        
        # Step 3: Group indices by the connected component of their occupant's value
        from collections import defaultdict
        root_to_indices = defaultdict(list)
        root_to_values = defaultdict(list)
        
        for i, val in enumerate(nums):
            root = uf.find(val_to_index[val])
            root_to_indices[root].append(i)
            root_to_values[root].append(val)
        
        # For each group, sort indices and occupant-values,
        # then assign in ascending order to get lexicographically smallest arrangement
        result = [0]*len(nums)
        for root in root_to_indices:
            idx_list = root_to_indices[root]
            val_list = root_to_values[root]
            idx_list.sort()
            val_list.sort()
            for i in range(len(idx_list)):
                result[idx_list[i]] = val_list[i]
        
        return result