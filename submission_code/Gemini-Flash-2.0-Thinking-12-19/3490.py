from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        def is_valid_subsequence(sub):
            if len(sub) < 2:
                return True
            first_sum_parity = (sub[0] + sub[1]) % 2
            for i in range(1, len(sub) - 1):
                if (sub[i] + sub[i+1]) % 2 != first_sum_parity:
                    return False
            return True
            
        if is_valid_subsequence(nums):
            return len(nums)
            
        even_count = 0
        odd_count = 0
        for num in nums:
            if num % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
                
        def get_alternating_length(start_parity): # 0 for even, 1 for odd
            length = 0
            last_index = -1
            expected_parity = start_parity
            for i in range(len(nums)):
                if nums[i] % 2 == expected_parity and i > last_index:
                    length += 1
                    last_index = i
                    expected_parity = 1 - expected_parity
            return length
            
        len_ae = get_alternating_length(0) # starting with even
        len_ao = get_alternating_length(1) # starting with odd
        
        return max(even_count, odd_count, len_ae, len_ao)