from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        b_nums = []
        for x in nums:
            if x % modulo == k:
                b_nums.append(1)
            else:
                b_nums.append(0)
        prefix_sum = 0
        prefix_sums = [-1] * n
        for i in range(n):
            prefix_sum += b_nums[i]
            prefix_sums[i] = prefix_sum
        
        interesting_subarray_count = 0
        prefix_sum_counts = {0: 1}
        
        for r in range(n):
            target_value = (prefix_sums[r] - k) % modulo
            if target_value in prefix_sum_counts:
                interesting_subarray_count += prefix_sum_counts[target_value]
            
            current_prefix_sum_mod = prefix_sums[r] % modulo
            if current_prefix_sum_mod in prefix_sum_counts:
                prefix_sum_counts[current_prefix_sum_mod] += 1
            else:
                prefix_sum_counts[current_prefix_sum_mod] = 1
                
        return interesting_subarray_count