from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute prefix OR array
        prefix_or_list = [0]
        current_or = 0
        for num in nums:
            current_or |= num
            prefix_or_list.append(current_or)
        
        # Compute suffix OR array
        suffix_or_list = [0] * (n + 1)
        suffix_or_list[n] = 0
        for i in range(n - 1, -1, -1):
            suffix_or_list[i] = nums[i] | suffix_or_list[i + 1]
        
        # full_or is the OR of all elements, which is prefix_or_list[n]
        max_value = prefix_or_list[n]
        
        # Now, for each i, compute or_without_i and the value after shifting i by k
        for i in range(n):
            # or_without_i is OR of elements before i and after i
            or_without_i = prefix_or_list[i] | suffix_or_list[i + 1]
            # Shift nums[i] by k
            shifted_num = (nums[i] << k)
            # The OR after shifting only i by k
            current_or_value = or_without_i | shifted_num
            # Update max_value
            if current_or_value > max_value:
                max_value = current_or_value
        
        return max_value