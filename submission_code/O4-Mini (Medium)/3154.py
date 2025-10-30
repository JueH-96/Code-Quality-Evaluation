from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        # max_before[j] will hold the maximum nums[i] for i in [0..j-1]
        max_before = [0] * n
        current_max = nums[0]
        for j in range(1, n):
            max_before[j] = current_max
            if nums[j] > current_max:
                current_max = nums[j]
        
        max_val = 0
        # iterate over j and k, use best i before j
        for j in range(1, n - 1):
            diff = max_before[j] - nums[j]
            # if diff <= 0, all (diff * nums[k]) <= 0, skip
            if diff <= 0:
                continue
            for k in range(j + 1, n):
                val = diff * nums[k]
                if val > max_val:
                    max_val = val
        
        return max_val