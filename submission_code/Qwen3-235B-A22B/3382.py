from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute log table for range maximum queries
        log_table = [0] * (n + 1)
        for i in range(2, n + 1):
            log_table[i] = log_table[i // 2] + 1
        
        # Build sparse table for range maximum queries
        k_max = log_table[n] + 1
        st = [[0] * n for _ in range(k_max)]
        st[0] = nums[:]
        for k in range(1, k_max):
            for i in range(n - (1 << k) + 1):
                st[k][i] = max(st[k-1][i], st[k-1][i + (1 << (k-1))])
        
        # Function to get maximum value in range [l, r]
        def get_max(l, r):
            if l > r:
                return -1
            length = r - l + 1
            k = log_table[length]
            return max(st[k][l], st[k][r - (1 << k) + 1])
        
        # Group positions by element value
        pos_dict = defaultdict(list)
        for idx, val in enumerate(nums):
            pos_dict[val].append(idx)
        
        total = 0
        
        # Process each element's positions
        for x in pos_dict:
            positions = pos_dict[x]
            m = len(positions)
            if m == 0:
                continue
            
            current_cluster_size = 1
            prev_pos = positions[0]
            
            for i in range(1, m):
                current_pos = positions[i]
                left = prev_pos + 1
                right = current_pos - 1
                
                if left > right:
                    # No elements in between, so extend the cluster
                    current_cluster_size += 1
                    prev_pos = current_pos
                    continue
                
                # Check the maximum in between
                max_val = get_max(left, right)
                if max_val <= x:
                    current_cluster_size += 1
                    prev_pos = current_pos
                else:
                    # Close current cluster and add to total
                    total += current_cluster_size * (current_cluster_size + 1) // 2
                    current_cluster_size = 1
                    prev_pos = current_pos
            
            # Add the last cluster
            total += current_cluster_size * (current_cluster_size + 1) // 2
        
        return total