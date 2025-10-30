from typing import List
import math

class Solution:
    
    # Helper class for Union-Find Disjoint Sets
    class UnionFind:
        def __init__(self, n):
            # Initialize parent array: parent[i] is the parent of element i
            # Initially, each element is its own parent
            self.parent = list(range(n))
            # Initialize size array for union by size optimization
            # size[i] stores the size of the tree rooted at i
            self.size = [1] * n
            # Initially, each element is in its own component
            self.num_components = n

        def find(self, i):
            # Find the root of the set containing element i
            # Path compression: make the found root the direct parent of i
            if self.parent[i] == i:
                return i
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

        def union(self, i, j):
            # Union the sets containing elements i and j
            root_i = self.find(i)
            root_j = self.find(j)

            # If they are already in the same set, do nothing
            if root_i != root_j:
                # Union by size: attach the root of the smaller tree to the root of the larger tree
                if self.size[root_i] < self.size[root_j]:
                    root_i, root_j = root_j, root_i
                
                self.parent[root_j] = root_i
                self.size[root_i] += self.size[root_j]
                
                # Decrease the number of components
                self.num_components -= 1
                return True # Union happened
            return False # Already in the same component

        def get_num_components(self):
            # Return the current number of disjoint sets (components)
            return self.num_components

    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        
        # 1. Separate nodes based on value > threshold or <= threshold
        # Nodes with value > threshold cannot be connected to any other node
        # (since lcm(a, b) >= max(a, b) > threshold for b > 0). Each forms a component of size 1.
        count_gt_threshold = 0
        indices_le_threshold = [] # Store original indices of nodes with value <= threshold
        value_to_index = {} # Map value -> original index for values <= threshold
        
        for i in range(n):
            if nums[i] > threshold:
                count_gt_threshold += 1
            else:
                indices_le_threshold.append(i)
                value_to_index[nums[i]] = i

        # If all nodes have value > threshold, return the count of such nodes
        if not indices_le_threshold:
            return count_gt_threshold

        # 2. Initialize Union-Find for nodes with value <= threshold
        # The size of the UF structure is the number of nodes with value <= threshold
        m = len(indices_le_threshold)
        uf = self.UnionFind(m) # Instantiate the inner UnionFind class
        
        # Map original index (index in nums) to the index within the UF structure (0 to m-1)
        original_to_uf_index = {original_idx: uf_idx for uf_idx, original_idx in enumerate(indices_le_threshold)}

        # 3. Iterate through potential common multiple values from 1 up to threshold
        # Two numbers a and b are connected if lcm(a, b) <= threshold.
        # This is equivalent to: there exists an integer L (1 <= L <= threshold)
        # such that a divides L and b divides L.
        # We iterate through each such L and find all numbers in nums (<= threshold)
        # that divide L. All these numbers must belong to the same connected component.
        for L in range(1, threshold + 1):
            # Find all divisors of L
            divisors_of_L = set()
            # Iterate up to sqrt(L) to find divisors efficiently
            # Using 1 to int(L**0.5) + 1 ensures we cover all pairs of factors
            for i in range(1, int(L**0.5) + 1):
                if L % i == 0:
                    divisors_of_L.add(i)
                    divisors_of_L.add(L // i)
            
            # Find which of these divisors are present in nums (with value <= threshold)
            # Collect their corresponding indices in the UF structure
            uf_indices_list = []
            for v in divisors_of_L:
                # Check if the divisor v is one of the values in nums <= threshold
                if v in value_to_index:
                    # Get the original index in nums for value v
                    original_idx = value_to_index[v]
                    # Get the corresponding index in the UF structure (0 to m-1)
                    uf_indices_list.append(original_to_uf_index[original_idx])
            
            # If we found at least two numbers in nums (<= threshold) that divide L,
            # they must be in the same component. Union them in the UF structure.
            # Union the first element with all subsequent elements in the list.
            # This ensures all found elements are connected to the first one,
            # placing them all in the same component.
            if len(uf_indices_list) >= 2:
                first_uf_idx = uf_indices_list[0]
                for j in range(1, len(uf_indices_list)):
                    uf.union(first_uf_idx, uf_indices_list[j])

        # 4. Calculate the total number of connected components
        # Total components = (components formed by nodes with value > threshold) +
        #                    (components formed by nodes with value <= threshold)
        total_components = count_gt_threshold + uf.get_num_components()

        return total_components