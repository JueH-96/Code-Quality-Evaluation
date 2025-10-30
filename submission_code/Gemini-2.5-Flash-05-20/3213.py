from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # Step 1: Find the maximum element in the array.
        # Constraints state 1 <= nums.length, so nums will not be empty.
        max_val = max(nums)

        # Step 2: Use a sliding window to count subarrays.
        ans = 0          # Total count of valid subarrays
        left = 0         # Left pointer of the sliding window
        count_max_val = 0 # Count of 'max_val' within the current window [left, right]
        n = len(nums)    # Length of the input array

        # Iterate with the 'right' pointer to expand the window
        for right in range(n):
            # If the current element at 'right' is the maximum value, increment its count
            if nums[right] == max_val:
                count_max_val += 1

            # While the current window [left, right] has at least 'k' occurrences of 'max_val':
            # This means the condition is met for this window.
            while count_max_val >= k:
                # If nums[left...right] is a valid subarray, then any subarray
                # starting at 'left' and ending at 'right' or any index further right
                # (up to n-1) will also be valid.
                # The number of such ending positions is (n - 1) - right + 1 = n - right.
                # Add these 'n - right' subarrays to our total count.
                ans += (n - right)

                # Now, shrink the window from the left to find other valid subarrays.
                # Move 'left' pointer to the right.
                # If the element leaving the window was 'max_val', decrement its count.
                if nums[left] == max_val:
                    count_max_val -= 1
                left += 1
        
        return ans