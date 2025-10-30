from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        """
        Calculates the minimum possible length of the array after performing
        pair removal operations.

        The strategy is to maximize the number of removed pairs. Since the
        array is sorted, a greedy two-pointer approach is effective. We pair
        elements from the first half of the array with elements from the second
        half.
        """
        n = len(nums)
        
        # We use two pointers: `i` for the first half and `j` for the second half.
        i = 0
        
        # Let the second half start from the middle index.
        # For n=5, mid=2. First half is [0,1], second half is [2,3,4].
        # For n=6, mid=3. First half is [0,1,2], second half is [3,4,5].
        mid = n // 2
        j = mid
        
        pairs = 0
        
        # The first half covers indices from 0 to mid-1.
        # The second half covers indices from mid to n-1.
        while i < mid and j < n:
            if nums[i] < nums[j]:
                # A valid pair (nums[i], nums[j]) is found.
                # We can remove this pair.
                pairs += 1
                i += 1
                j += 1
            else:
                # nums[i] >= nums[j]. This pair is not valid.
                # To find a valid partner for nums[i], we need a larger element
                # from the second half. So, we advance `j`.
                # The element nums[j] is effectively left unpaired because it's too
                # small for nums[i] and any subsequent elements in the first half.
                j += 1
                
        # Each pair removal reduces the length by 2.
        return n - (2 * pairs)