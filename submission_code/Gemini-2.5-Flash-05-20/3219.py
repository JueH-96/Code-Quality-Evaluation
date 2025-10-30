from collections import defaultdict
from typing import List

class DSU:
    def __init__(self, n):
        # parent[i] stores the parent of element i
        self.parent = list(range(n))
        # rank[i] stores the rank (or size for union by size) of the tree rooted at i.
        # Used for optimization during union operation to keep trees flat.
        self.rank = [0] * n 

    def find(self, i):
        # Path compression: if i is not its own parent, set its parent to the root of its set.
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by rank: attach the smaller rank tree under the root of the larger rank tree.
            # This helps keep the overall tree height small.
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                # If ranks are equal, pick one as root and increment its rank.
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True # Successfully united
        return False # Already in the same set

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        dsu = DSU(n)

        # 1. Create (value, original_index) pairs
        # This preserves the original index for each number.
        val_idx_pairs = []
        for i in range(n):
            val_idx_pairs.append((nums[i], i))

        # 2. Sort pairs by value.
        # This step is crucial. It brings "connectable" numbers (those whose values are
        # within `limit` of each other) into adjacent positions or close proximity,
        # making it easy to identify potential connections.
        val_idx_pairs.sort()

        # 3. Iterate through sorted pairs and union connected components using DSU.
        # If val2 - val1 <= limit, then idx1 and idx2 can be considered connected.
        # Transitive property ensures all elements in a component are linked.
        for k in range(n - 1):
            val1, idx1 = val_idx_pairs[k]
            val2, idx2 = val_idx_pairs[k+1]
            if val2 - val1 <= limit: # Values are already sorted, so val2 >= val1
                dsu.union(idx1, idx2)

        # 4. Group original indices and their corresponding values by connected component.
        # Each 'root' in the DSU represents a unique connected component.
        # component_map: {root_index -> {'indices': [original_indices_in_component], 'values': [original_values_in_component]}}
        component_map = defaultdict(lambda: {'indices': [], 'values': []})

        for i in range(n):
            root = dsu.find(i) # Find the representative (root) of the set that index 'i' belongs to.
            component_map[root]['indices'].append(i)
            component_map[root]['values'].append(nums[i])
        
        # 5. Construct the result array by placing sorted values into sorted original positions
        # within each connected component to achieve lexicographical smallest order.
        ans = [0] * n # Initialize the result array with placeholder values.

        for root in component_map:
            indices = component_map[root]['indices']
            values = component_map[root]['values']

            # To achieve the lexicographically smallest arrangement, we must place
            # the smallest values into the smallest available original indices within this component.
            indices.sort() # Sort the original indices in ascending order.
            values.sort()  # Sort the values in ascending order.

            # Assign sorted values to sorted indices.
            for i in range(len(indices)):
                ans[indices[i]] = values[i]
        
        return ans