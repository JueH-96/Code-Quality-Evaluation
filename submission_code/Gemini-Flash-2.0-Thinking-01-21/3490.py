from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_nums = []
        odd_nums = []
        for num in nums:
            if num % 2 == 0:
                even_nums.append(num)
            else:
                odd_nums.append(num)
        len_even_subsequence = len(even_nums)
        len_odd_subsequence = len(odd_nums)
        
        eo_subsequence = []
        expected_parity = 0 # 0 for even, 1 for odd
        last_index = -1
        for i in range(len(nums)):
            num = nums[i]
            if num % 2 == expected_parity:
                eo_subsequence.append(num)
                expected_parity = 1 - expected_parity
                
        oe_subsequence = []
        expected_parity = 1 # 1 for odd, 0 for even
        last_index = -1
        for i in range(len(nums)):
            num = nums[i]
            if num % 2 == expected_parity:
                oe_subsequence.append(num)
                expected_parity = 1 - expected_parity
                
        len_eo_subsequence = len(eo_subsequence)
        len_oe_subsequence = len(oe_subsequence)
        
        return max(len_even_subsequence, len_odd_subsequence, len_eo_subsequence, len_oe_subsequence)