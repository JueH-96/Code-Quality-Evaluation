import math
from typing import List

class DSU:
    def __init__(self, n_elements):
        # parent[i] stores the parent of element i
        self.parent = list(range(n_elements))
        # ranks[i] stores the rank of the tree rooted at i (for union by rank optimization)
        self.ranks = [0] * n_elements
        # num_sets tracks the current number of disjoint sets/components
        self.num_sets = n_elements 

    def find(self, i):
        # Path compression: if i is not the root, set its parent to the root of its set
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        # If roots are different, they are in different sets, so merge them
        if root_i != root_j:
            # Union by rank: attach the smaller rank tree under the root of the larger rank tree
            if self.ranks[root_i] < self.ranks[root_j]:
                self.parent[root_i] = root_j
            elif self.ranks[root_j] < self.ranks[root_i]:
                self.parent[root_j] = root_i
            else:
                # If ranks are equal, attach one to the other and increment the rank of the new root
                self.parent[root_j] = root_i
                self.ranks[root_i] += 1
            self.num_sets -= 1 # One less component after successful union
            return True # Union happened
        return False # Already in the same set

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        isolated_large_values_count = 0
        small_nums_values = []
        # val_to_dsu_idx maps an actual number value (e.g., 2, 4, 8) to its corresponding
        # index in the DSU's internal array (0, 1, 2, etc.)
        val_to_dsu_idx = {} 

        # Step 1: Separate numbers > threshold (isolated) from numbers <= threshold (potentially connected)
        for x in nums:
            if x <= threshold:
                # Ensure unique values from nums are processed only once and get unique DSU indices
                if x not in val_to_dsu_idx:
                    val_to_dsu_idx[x] = len(small_nums_values)
                    small_nums_values.append(x)
            else:
                isolated_large_values_count += 1
        
        # Step 2: Initialize DSU for numbers <= threshold
        n_small_nums = len(small_nums_values)
        dsu = DSU(n_small_nums)

        # Step 3: Sieve-like approach to group numbers by their common multiples up to threshold.
        # `divisors_at_multiple[k]` will store a list of values from `small_nums_values`
        # that divide `k`.
        divisors_at_multiple = [[] for _ in range(threshold + 1)]

        # Populate divisors_at_multiple: For each `val` in `small_nums_values`,
        # add `val` to the lists for all its multiples up to `threshold`.
        for val in small_nums_values:
            for multiple in range(val, threshold + 1, val):
                divisors_at_multiple[multiple].append(val)
        
        # Step 4: Iterate through all possible common multiples `k` from 1 to `threshold`.
        # If multiple values from `small_nums_values` divide `k`, they are connected.
        for k in range(1, threshold + 1):
            connected_group = divisors_at_multiple[k]
            if len(connected_group) > 1:
                # All values in `connected_group` divide `k`. This implies their `lcm` also divides `k`.
                # Since `k <= threshold`, their `lcm` is also `<= threshold`.
                # Thus, all values in this group are connected.
                # Union them all together using the first value as the representative.
                first_val = connected_group[0]
                first_idx = val_to_dsu_idx[first_val]
                for i in range(1, len(connected_group)):
                    current_val = connected_group[i]
                    current_idx = val_to_dsu_idx[current_val]
                    dsu.union(first_idx, current_idx)
        
        # Step 5: Calculate the total number of components.
        # This is the sum of:
        # a. Components formed by values <= threshold (tracked by DSU's num_sets).
        # b. Components formed by values > threshold (these were counted as isolated).
        return dsu.num_sets + isolated_large_values_count