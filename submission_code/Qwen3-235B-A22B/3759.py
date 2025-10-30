from bisect import bisect_left
from collections import defaultdict

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        elements = list(zip(nums1, nums2, range(n)))
        elements.sort()  # Sort by nums1 values
        
        # Coordinate compression for nums2 values
        all_vals = nums2.copy()
        sorted_unique = sorted(set(all_vals))
        rank_dict = {v: i+1 for i, v in enumerate(sorted_unique)}  # 1-based indexing
        m = len(sorted_unique)
        
        # Fenwick Tree implementation for count and sum
        class Fenwick:
            def __init__(self, size):
                self.size = size
                self.tree_cnt = [0] * (self.size + 2)
                self.tree_sum = [0] * (self.size + 2)
            
            def update(self, rank, value):
                # Add an element with given rank and value
                while rank <= self.size:
                    self.tree_cnt[rank] += 1
                    self.tree_sum[rank] += value
                    rank += rank & -rank
            
            def query(self, rank):
                # Returns (count, sum) up to given rank
                cnt = 0
                sm = 0
                while rank > 0:
                    cnt += self.tree_cnt[rank]
                    sm += self.tree_sum[rank]
                    rank += rank & -rank
                return (cnt, sm)
        
        ft = Fenwick(m)
        res = [0] * n
        i = 0
        
        # Process each group in the sorted elements list
        while i < n:
            current_val = elements[i][0]
            j = i
            current_group = []
            # Collect all elements in the current group (same nums1 value)
            while j < n and elements[j][0] == current_val:
                current_group.append(elements[j])
                j += 1
            
            # Query the Fenwick tree to get the sum of top k elements so far
            remaining = k
            current_sum = 0
            # Iterate over all unique values in descending order
            for v in reversed(sorted_unique):
                r = rank_dict[v]
                # Get count and sum up to r
                cnt_prev, sm_prev = ft.query(r - 1)
                cnt_r, sm_r = ft.query(r)
                current_c = cnt_r - cnt_prev
                current_s = sm_r - sm_prev
                
                if current_c == 0:
                    continue  # Skip if no elements here
                
                if current_c <= remaining:
                    current_sum += current_s
                    remaining -= current_c
                else:
                    current_sum += v * remaining
                    remaining = 0
                
                if remaining == 0:
                    break
            
            # Assign the current_sum to all elements in the current group
            for (_, val2, idx) in current_group:
                res[idx] = current_sum if remaining >= 0 else 0
            
            # Update the Fenwick tree with the current group's nums2 values
            for (_, val2, _) in current_group:
                r = rank_dict[val2]
                ft.update(r, val2)
            
            i = j  # Move to the next group
        
        return res