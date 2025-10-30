import math
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        """
        Calculates the minimum possible length of the array after performing
        the described operations any number of times.
        """
        
        # Find the minimum element in the array.
        min_val = min(nums)
        
        # Check if the minimum element divides all other elements.
        # If we find any element `num` such that `num % min_val != 0`, then the
        # GCD of the array is smaller than `min_val`.
        # In this case, we can always reduce the array to a single element.
        for num in nums:
            if num % min_val != 0:
                return 1
                
        # If all elements are divisible by `min_val`, then `min_val` is the GCD.
        # We count the number of occurrences of this minimum value.
        count_min = nums.count(min_val)
        
        # The minimum length achievable is ceil(count_min / 2).
        # This can be calculated using integer arithmetic as (count_min + 1) // 2 for positive integers.
        return (count_min + 1) // 2