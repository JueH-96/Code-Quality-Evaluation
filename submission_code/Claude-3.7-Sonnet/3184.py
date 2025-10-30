from sortedcontainers import SortedDict

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        diff = [nums[i] - i for i in range(n)]
        
        # best_sum[diff_value] = maximum sum of a balanced subsequence ending with diff_value
        best_sum = SortedDict()
        
        for i, num in enumerate(nums):
            d = diff[i]
            
            # Start a new subsequence or extend an existing one
            curr_max = num
            
            # Find the maximum sum for all diff <= d
            idx = best_sum.bisect_right(d)
            if idx > 0:
                prev_diff = best_sum.keys()[idx - 1]
                curr_max = max(curr_max, best_sum[prev_diff] + num)
            
            # Update best_sum with the new maximum
            if d in best_sum:
                best_sum[d] = max(best_sum[d], curr_max)
            else:
                best_sum[d] = curr_max
            
            # Remove all diff > d with a smaller sum
            # This is an optimization: if a larger diff has a smaller sum, it's never beneficial
            to_remove = []
            idx = best_sum.bisect_right(d)
            while idx < len(best_sum):
                next_diff = best_sum.keys()[idx]
                if best_sum[next_diff] <= curr_max:
                    to_remove.append(next_diff)
                idx += 1
            
            for diff_val in to_remove:
                del best_sum[diff_val]
        
        return max(best_sum.values()) if best_sum else max(nums)