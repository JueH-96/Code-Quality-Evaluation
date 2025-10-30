from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_positive_sum = float('inf')

        # Iterate through all possible starting indices of a subarray.
        for i in range(n):
            current_sum = 0
            # From each starting index, iterate through all possible ending indices.
            for j in range(i, n):
                # Efficiently update the sum of the subarray nums[i..j].
                current_sum += nums[j]
                
                # Calculate the length of the current subarray.
                length = j - i + 1

                # Check if the subarray meets the length requirement.
                if l <= length <= r:
                    # If the subarray's sum is positive, it's a valid candidate.
                    if current_sum > 0:
                        # Update the minimum positive sum found so far.
                        min_positive_sum = min(min_positive_sum, current_sum)
                        
        # If min_positive_sum remains at its initial value, no subarray
        # satisfied the conditions. Return -1 as required.
        # Otherwise, return the minimum positive sum found.
        return min_positive_sum if min_positive_sum != float('inf') else -1