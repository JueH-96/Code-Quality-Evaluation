from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Step 1: Precompute smallest prime factors up to 10^6
        MAX = 10**6 + 1
        spf = [0] * MAX
        for i in range(2, MAX):
            if spf[i] == 0:
                spf[i] = i
                for j in range(i * 2, MAX, i):
                    if spf[j] == 0:
                        spf[j] = i
        
        # Step 2: For each x in nums, determine possible values and operations
        possible = []
        for x in nums:
            options = []
            options.append((x, 0))
            if x > 1:
                if spf[x] != x:  # Composite number
                    reduced = x // spf[x]
                    options.append((reduced, 1))
            possible.append(options)
        
        # Step 3: Collect all unique possible values and compress them
        unique_values = set()
        for opts in possible:
            for v, _ in opts:
                unique_values.add(v)
        sorted_values = sorted(unique_values)
        value_to_idx = {v: i + 1 for i, v in enumerate(sorted_values)}
        size = len(sorted_values) + 1  # +1 for sentinel
        
        # Fenwick Tree for range minimum queries and point updates
        class FenwickTree:
            def __init__(self, size):
                self.N = size + 2  # To avoid issues with indices
                self.tree = [float('inf')] * (self.N)
            
            def update(self, idx, value):
                while idx < self.N:
                    if value < self.tree[idx]:
                        self.tree[idx] = value
                    else:
                        break  # No need to update further if current value is larger
                    idx += idx & -idx
            
            def query(self, idx):
                res = float('inf')
                while idx > 0:
                    if self.tree[idx] < res:
                        res = self.tree[idx]
                    idx -= idx & -idx
                return res
        
        # Initialize Fenwick Tree
        prev_fen = FenwickTree(size)
        # Initialize with sentinel
        prev_fen.update(0, 0)
        # Set operations for the first element
        v_idx_offset = 1  # To skip index 0
        for v, o in possible[0]:
            idx = value_to_idx[v]
            min_op = prev_fen.query(idx)
            if min_op != float('inf'):
                total_op = min_op + o
                prev_fen.update(idx, total_op)
        
        # Process remaining elements
        for i in range(1, len(nums)):
            current_fen = FenwickTree(size)
            for v, o in possible[i]:
                idx = value_to_idx[v]
                min_op = prev_fen.query(idx)
                if min_op != float('inf'):
                    total_op = min_op + o
                    current_fen.update(idx, total_op)
            prev_fen = current_fen
        
        # Find the minimal operations from the final Fenwick Tree
        min_total_op = float('inf')
        for v in sorted_values:
            idx = value_to_idx[v]
            min_total_op = min(min_total_op, prev_fen.query(idx))
        
        return min_total_op if min_total_op != float('inf') else -1