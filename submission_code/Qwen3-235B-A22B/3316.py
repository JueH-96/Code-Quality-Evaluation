from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        if n < k:
            return 0
        
        # Generate all pairwise differences
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(sorted_nums[j] - sorted_nums[i])
        sorted_diffs = sorted(diffs)
        m = len(sorted_diffs)
        if m == 0:
            return 0
        
        # Precompute f(d) for all d in sorted_diffs
        f_dict = {}
        for d in sorted_diffs:
            f_val = self.compute_f(sorted_nums, k, d)
            f_dict[d] = f_val
        
        # Calculate the total sum
        total = 0
        for i in range(m):
            current_d = sorted_diffs[i]
            current_f = f_dict.get(current_d, 0)
            if i < m - 1:
                next_d = sorted_diffs[i + 1]
                next_f = f_dict.get(next_d, 0)
            else:
                next_f = 0
            contribution = current_d * (current_f - next_f)
            total += contribution
        
        return total % MOD

    def compute_f(self, sorted_nums: List[int], k: int, d: int) -> int:
        n = len(sorted_nums)
        if k == 0 or n < k:
            return 0
        
        # Initialize dp and prefix sums for c=1
        pre_c = [0] * n
        dp = [ [0] * (k + 1) for _ in range(n)]
        for i in range(n):
            dp[i][1] = 1
        # Compute prefix sums for c=1
        pre_c[0] = dp[0][1]
        for i in range(1, n):
            pre_c[i] = pre_c[i - 1] + dp[i][1]
        
        if k == 1:
            return pre_c[-1]
        
        # Iterate for c from 2 to k
        for c in range(2, k + 1):
            curr_dp = [0] * n
            for i in range(n):
                target = sorted_nums[i] - d
                low, high = 0, i - 1
                idx = -1
                while low <= high:
                    mid = (low + high) // 2
                    if sorted_nums[mid] <= target:
                        idx = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                if idx >= 0:
                    curr_dp[i] = pre_c[idx]
                else:
                    curr_dp[i] = 0
            # Update curr_pre
            curr_pre = [0] * n
            curr_pre[0] = curr_dp[0]
            for i in range(1, n):
                curr_pre[i] = curr_pre[i - 1] + curr_dp[i]
            pre_c = curr_pre
        
        return pre_c[-1]