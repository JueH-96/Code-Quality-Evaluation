from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        min_length = float('inf')  # Initialize with positive infinity
        n = len(nums)

        # Constraints from the problem statement:
        # 1 <= nums.length <= 50
        # 0 <= nums[i] <= 50
        # 0 <= k < 64

        # Handle the k = 0 case explicitly.
        # Any non-empty subarray has an OR sum >= 0 (since nums[i] >= 0).
        # The shortest non-empty subarray has length 1 (e.g., [nums[0]]).
        # Its OR sum is nums[0]. Since nums[0] >= 0, it's >= k (if k=0).
        if k == 0:
            return 1

        # Iterate over all possible start points of a subarray
        for i in range(n):
            current_or_sum = 0
            # Iterate over all possible end points of a subarray, starting from i
            for j in range(i, n):
                # Update the OR sum for the current subarray nums[i...j]
                current_or_sum |= nums[j]
                
                # Check if the current subarray is special
                if current_or_sum >= k:
                    # If it is, calculate its length
                    length = j - i + 1
                    # Update the minimum length found so far
                    min_length = min(min_length, length)
                    
                    # Optimization: For a fixed starting point i, nums[i...j] is the
                    # first (and thus shortest) subarray that satisfies the condition.
                    # Any further extension (nums[i...j+1], etc.) will also satisfy
                    # the OR sum condition (as OR sum is non-decreasing) but will be longer.
                    # So, we can break from this inner loop for j.
                    break
        
        # If min_length is still infinity, no special subarray was found
        if min_length == float('inf'):
            return -1
        else:
            # Otherwise, return the minimum length found.
            # Cast to int as min_length might be a float (e.g., 1.0).
            return int(min_length)