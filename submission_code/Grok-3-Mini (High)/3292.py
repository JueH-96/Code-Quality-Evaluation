from typing import List
import bisect

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Check if all indices appear at least once in changeIndices
        indices_set = set(changeIndices)
        if len(indices_set) < n:
            return -1
        
        def can_do(t: int) -> bool:
            # Check if all indices appear in first t seconds
            appear = set()
            for s in range(1, t + 1):  # s from 1 to t
                idx = changeIndices[s - 1]
                appear.add(idx)
            if len(appear) < n:
                return False
            
            # Assign the maximum s_i <= t for each index i
            mark_time = [0] * (n + 1)  # 1-based, store s for each index
            for i in range(1, n + 1):
                for s in range(t, 0, -1):  # s from t to 1
                    if changeIndices[s - 1] == i:
                        mark_time[i] = s
                        break  # assign the largest s
            
            # Compute D_i = mark_time[i] - 1 for each i
            D_val = [mark_time[i] - 1 for i in range(1, n + 1)]  # length n, for indices 1 to n
            
            # Create pairs (D, nums) for each index (nums is 0-based)
            pairs = [(D_val[j], nums[j]) for j in range(n)]  # j from 0 to n-1
            
            # Sort pairs by D ascending
            sorted_pairs = sorted(pairs, key=lambda x: x[0])
            
            # Extract sorted D and nums
            D_sorted = [p[0] for p in sorted_pairs]
            nums_sorted = [p[1] for p in sorted_pairs]
            
            # Compute prefix sum of nums_sorted
            prefix_sum_nums = [0]
            for num in nums_sorted:
                prefix_sum_nums.append(prefix_sum_nums[-1] + num)
            
            # Check for each tau from 0 to t
            for tau in range(0, t + 1):
                # Number with D_i <= tau
                num_leq_tau = bisect.bisect_right(D_sorted, tau)
                G_tau = prefix_sum_nums[num_leq_tau]
                
                # F(tau - 1): number with D_i <= tau - 1
                if tau - 1 >= 0:
                    F_tau_minus_1 = bisect.bisect_right(D_sorted, tau - 1)
                else:
                    F_tau_minus_1 = 0  # for tau -1 = -1
                
                avail_tau = tau - F_tau_minus_1
                if G_tau > avail_tau:
                    return False
            
            return True
        
        # Binary search for the smallest t
        left, right = 1, m
        while left <= right:
            mid = (left + right) // 2
            if can_do(mid):
                right = mid - 1
            else:
                left = mid + 1
        
        # After binary search, left is the smallest t where can_do is True, or m+1
        if left > m:
            return -1
        else:
            return left