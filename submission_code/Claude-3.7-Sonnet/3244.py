import math

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_val = min(nums)
        
        # Check if all elements are divisible by the minimum value
        all_divisible = True
        for num in nums:
            if num % min_val != 0:
                all_divisible = False
                break
        
        if all_divisible:
            # If all elements are divisible by min_val, we can't get any element
            # smaller than min_val. The best we can do is reduce pairs of min_val
            # to 0s, resulting in ceiling(count_min/2) elements.
            count_min = nums.count(min_val)
            return math.ceil(count_min / 2)
        else:
            # If not all elements are divisible by min_val, we can always get
            # a smaller element through the operation. Through repeated operations,
            # we can reduce the array to a single element.
            return 1