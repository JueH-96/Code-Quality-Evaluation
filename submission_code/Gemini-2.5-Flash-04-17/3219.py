from typing import List
from collections import defaultdict

class DSU:
    def __init__(self, n):
        # parent[i] is the parent of node i
        self.parent = list(range(n))
        # size[i] is the size of the set rooted at i
        self.size = [1] * n

    def find(self, i):
        """Finds the representative of the set containing element i."""
        if self.parent[i] == i:
            return i
        # Path compression
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Unites the set containing element i and the set containing element j."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by size
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True  # Union happened
        return False # Already in the same set


class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        
        # Pair each number with its original index
        # This allows us to sort by value while keeping track of the original position
        indexed_nums = sorted((num, i) for i, num in enumerate(nums))
        
        # Initialize Disjoint Set Union structure for n elements (original indices)
        dsu = DSU(n)
        
        # Iterate through the sorted numbers. If two adjacent numbers are within the limit,
        # their original indices belong to the same connected component.
        # The transitive property of DSU handles chains of such connections.
        for i in range(n - 1):
            # indexed_nums[i][0] is the value, indexed_nums[i][1] is the original index
            if indexed_nums[i+1][0] - indexed_nums[i][0] <= limit:
                dsu.union(indexed_nums[i][1], indexed_nums[i+1][1])
                
        # Group original indices and their corresponding original values by their connected component root
        # components_map[root] will store {'indices': [...], 'values': [...]}
        components_map = defaultdict(lambda: {'indices': [], 'values': []})
        for i in range(n):
            root = dsu.find(i)
            components_map[root]['indices'].append(i)
            components_map[root]['values'].append(nums[i]) # Use original nums[i]
            
        # Build the result array
        result = [0] * n
        
        # For each connected component
        for root, data in components_map.items():
            indices_in_component = data['indices']
            values_in_component = data['values']
            
            # To make the array lexicographically smallest, we must place the smallest
            # available values into the smallest available indices within the component.
            
            # Sort the original indices within the component
            indices_in_component.sort()
            # Sort the original values within the component
            values_in_component.sort()
            
            # Assign the sorted values to the sorted indices
            for k in range(len(indices_in_component)):
                result[indices_in_component[k]] = values_in_component[k]
        
        return result