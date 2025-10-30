from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even = 0
        odd = 0
        for num in nums:
            if num % 2 == 0:
                even += 1
            else:
                odd += 1
        type_a = max(even, odd)
        
        parity_changes = 0
        if not nums:
            return 0
        prev_parity = nums[0] % 2
        for num in nums[1:]:
            curr_parity = num % 2
            if curr_parity != prev_parity:
                parity_changes += 1
            prev_parity = curr_parity
        type_b = parity_changes + 1
        
        return max(type_a, type_b)