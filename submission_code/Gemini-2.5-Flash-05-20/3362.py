import collections
from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Calculate the total number of subarrays.
        # For an array of length n, there are n*(n+1)/2 subarrays.
        total_subarrays = n * (n + 1) // 2
        
        # The 0-indexed rank of the median element in the sorted uniqueness array.
        # If the total count is M, the median is at index (M-1)//2 to get
        # the smaller of the two middle elements for an even M.
        target_rank = (total_subarrays - 1) // 2

        # This helper function counts the number of subarrays nums[i..j] such that
        # the number of distinct elements in nums[i..j] is less than or equal to x.
        def count_le(x: int) -> int:
            count = 0
            # freq stores the frequency of elements in the current sliding window [left, right-1]
            freq = collections.defaultdict(int)
            distinct_count = 0
            right = 0  # The right pointer for the sliding window

            # Iterate through all possible left endpoints of the subarrays
            for left in range(n):
                # Expand the window from the right until distinct_count > x
                while right < n:
                    val = nums[right]
                    # If nums[right] is a new distinct element in the current window
                    if freq[val] == 0:
                        # If adding this new distinct element would make distinct_count exceed x,
                        # then the current window [left, right-1] is the longest valid window
                        # starting at `left`. We stop expanding for this `left`.
                        if distinct_count == x:
                            break 
                        distinct_count += 1
                    
                    freq[val] += 1 # Increment frequency of nums[right]
                    right += 1 # Move right pointer forward
                
                # At this point, the window nums[left .. right-1] is the longest subarray
                # starting at `left` that has `distinct_count <= x`.
                # All subarrays from [left, left] to [left, right-1] satisfy the condition.
                # The number of such subarrays is (right - 1) - left + 1 = right - left.
                count += (right - left)

                # Before moving to the next `left` (i.e., shrinking the window from the left),
                # decrement the frequency of nums[left] and update distinct_count if necessary.
                val_to_remove = nums[left]
                freq[val_to_remove] -= 1
                if freq[val_to_remove] == 0:
                    distinct_count -= 1
            
            return count

        # Binary search for the median value in the range [1, n].
        # The number of distinct elements in any subarray is at least 1 and at most n.
        low = 1
        high = n
        ans = n  # Initialize ans with a possible maximum value (n)

        while low <= high:
            mid = low + (high - low) // 2
            
            # Count how many subarrays have distinct elements less than or equal to `mid`.
            num_subarrays_le_mid = count_le(mid)
            
            # If the count of such subarrays is greater than target_rank,
            # it means `mid` could be the median or the median is smaller.
            # We record `mid` as a potential answer and try to find a smaller one.
            if num_subarrays_le_mid > target_rank:
                ans = mid
                high = mid - 1
            else:
                # If the count is less than or equal to target_rank,
                # `mid` is too small, so we need to look in the larger half.
                low = mid + 1
        
        return ans