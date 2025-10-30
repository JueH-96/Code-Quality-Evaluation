import math
from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_abs_diff = float('inf')
        
        # `available_elements` will store nums[k] values for indices k
        # such that k <= current_j - x. These are elements that are
        # at least 'x' indices to the left of the current element nums[current_j].
        # Using SortedList allows efficient O(log N) addition and
        # O(log N) search for the closest elements.
        available_elements = SortedList()
        
        # Iterate through the array using 'j' as the current right index.
        # We are looking for a pair (i, j) such that abs(i - j) >= x.
        # This loop effectively considers all pairs where i < j and j - i >= x.
        # The symmetric case (j < i and i - j >= x) is covered because
        # when the larger index (i) becomes the 'j' in the loop, the smaller index (j)
        # would have already been added to 'available_elements' (if j <= i - x).
        for j in range(n):
            # When j reaches a point where nums[j - x] is a valid index,
            # this element nums[j - x] becomes a candidate for nums[i].
            # For this nums[j - x], its index (j - x) satisfies:
            # abs(j - (j - x)) = x, which meets the condition abs(idx_current - idx_candidate) >= x.
            if j - x >= 0:
                available_elements.add(nums[j - x])
            
            # If there are elements in `available_elements`, it means we have
            # candidate nums[i] values (where i <= j - x) that can be paired with nums[j].
            if available_elements:
                current_num = nums[j]
                
                # Find the position where `current_num` would be inserted in the sorted list.
                # Elements at `idx` and `idx - 1` are the closest to `current_num`.
                idx = available_elements.bisect_left(current_num)
                
                # Check the element at `idx` (the smallest element >= current_num)
                if idx < len(available_elements):
                    min_abs_diff = min(min_abs_diff, abs(available_elements[idx] - current_num))
                
                # Check the element at `idx - 1` (the largest element <= current_num)
                if idx > 0:
                    min_abs_diff = min(min_abs_diff, abs(available_elements[idx - 1] - current_num))
                    
        return min_abs_diff