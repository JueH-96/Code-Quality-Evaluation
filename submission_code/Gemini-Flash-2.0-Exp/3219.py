class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        indices = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_x] = root_y
        
        parent = list(range(n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(nums[i] - nums[j]) <= limit:
                    union(i, j)
        
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        result = [0] * n
        for root in groups:
            group_indices = groups[root]
            group_values = [nums[i] for i in group_indices]
            group_indices.sort()
            group_values.sort()
            
            for i in range(len(group_indices)):
                result[group_indices[i]] = group_values[i]
        
        return result