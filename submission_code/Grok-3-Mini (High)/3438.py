from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        N = len(nums)
        # Initialize peak_status array
        peak_status = [0] * N
        for i in range(1, N - 1):  # i from 1 to N-2 inclusive
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                peak_status[i] = 1
        
        # Initialize Fenwick Tree
        ft = [0] * N  # Size N, indices 1 to N-2 used
        
        # Define nested functions for Fenwick Tree operations
        def update_ft(idx, delta):
            idx_copy = idx
            while idx_copy < N:
                ft[idx_copy] += delta
                idx_copy += idx_copy & -idx_copy
        
        def prefix_sum(idx):
            sum_res = 0
            i = idx
            while i > 0:
                sum_res += ft[i]
                i -= i & -i
            return sum_res
        
        # Set initial values in Fenwick Tree
        for i in range(1, N - 1):
            update_ft(i, peak_status[i])
        
        # Process queries
        ans = []
        for query in queries:
            op = query[0]
            if op == 2:  # Update operation
                index = query[1]
                val = query[2]
                # Update nums[index] to val
                nums[index] = val
                # Update affected peak statuses
                k = index
                for di in [-1, 0, 1]:
                    cand_i = k + di
                    if 1 <= cand_i <= N - 2:  # cand_i must be within 1 to N-2
                        # Recompute new peak value
                        if nums[cand_i] > nums[cand_i - 1] and nums[cand_i] > nums[cand_i + 1]:
                            new_peak = 1
                        else:
                            new_peak = 0
                        old_peak = peak_status[cand_i]
                        if new_peak != old_peak:
                            delta = new_peak - old_peak
                            update_ft(cand_i, delta)
                            peak_status[cand_i] = new_peak
            elif op == 1:  # Query operation
                l = query[1]
                r = query[2]
                low = max(l + 1, 1)
                high = min(r - 1, N - 2)
                if low <= high:
                    count = prefix_sum(high) - prefix_sum(low - 1)
                else:
                    count = 0
                ans.append(count)
        
        return ans