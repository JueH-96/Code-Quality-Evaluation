import bisect

class Solution:
    def sumOfPowers(self, nums: list, k: int) -> int:
        MOD = 10**9 + 7
        if k == 0:
            return 0
        n = len(nums)
        if n < k:
            return 0
        
        # Sort the array
        sorted_nums = sorted(nums)
        
        # Generate all unique pairwise differences
        differences = set()
        for i in range(n):
            for j in range(i + 1, n):
                differences.add(sorted_nums[j] - sorted_nums[i])
        D = sorted(differences)
        
        counts = {}
        
        for d in D:
            # Compute the number of k-sized subsets where all elements are at least d apart
            # Using optimized 1D DP with prefix sums
            dp_prev = [0] * n
            for i in range(n):
                dp_prev[i] = 1  # m=1 case
            
            if k == 1:
                count_d = sum(dp_prev)
                counts[d] = count_d
                continue
            
            for m in range(2, k + 1):
                # Compute prefix sums for the previous m-1
                prefix = [0] * n
                prefix[0] = dp_prev[0]
                for i in range(1, n):
                    prefix[i] = prefix[i - 1] + dp_prev[i]
                
                current_dp = [0] * n
                for i in range(n):
                    target = sorted_nums[i] - d
                    j = bisect.bisect_right(sorted_nums, target) - 1
                    if j >= 0:
                        sum_prev = prefix[j]
                    else:
                        sum_prev = 0
                    current_dp[i] = sum_prev
                
                dp_prev = current_dp
            
            count_d = sum(dp_prev)
            counts[d] = count_d
        
        total = 0
        len_D = len(D)
        for i in range(len_D):
            current_d = D[i]
            if i < len_D - 1:
                next_d = D[i + 1]
                cnt_next = counts.get(next_d, 0)
            else:
                cnt_next = 0
            cnt_current = counts.get(current_d, 0)
            contribution = (cnt_current - cnt_next) * current_d
            total += contribution
        
        return total % MOD