class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        dp_sum = {}
        dp_count = {}
        mod = 10**9 + 7
        for i in range(len(nums)):
            num = nums[i]
            current_sum = num
            current_count = 1
            for j in range(i):
                prev_num = nums[j]
                if abs(num - prev_num) == 1:
                    prev_end_sum = dp_sum.get((j, prev_num), 0)
                    prev_end_count = dp_count.get((j, prev_num), 0)
                    current_sum = (current_sum + prev_end_sum + (prev_end_count * num) % mod) % mod
                    current_count = (current_count + prev_end_count) % mod
            dp_sum[(i, num)] = current_sum
            dp_count[(i, num)] = current_count
        
        final_sum = 0
        for key in dp_sum:
            final_sum = (final_sum + dp_sum[key]) % mod
            
        all_single_element_subsequence_sum = 0
        for num in nums:
            all_single_element_subsequence_sum = (all_single_element_subsequence_sum + num) % mod
            
        return final_sum