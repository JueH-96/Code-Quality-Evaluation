from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        total_count = 0
        # current_count_ending_here stores the number of alternating subarrays ending at the current index i.
        # This is equivalent to the length of the longest alternating subarray ending at index i.
        current_count_ending_here = 0

        for i in range(n):
            # Check if the current element breaks the alternating pattern with the previous element
            # This check is only relevant for i > 0
            if i > 0 and nums[i] == nums[i-1]:
                # If the pattern is broken (adjacent elements are the same),
                # the alternating subarray ending at the current index can only be the element itself.
                # So, the count of alternating subarrays ending here resets to 1.
                current_count_ending_here = 1
            else: # i == 0 or nums[i] != nums[i-1]
                # If it's the first element (i==0), the alternating subarray is [nums[0]], count 1.
                # If the current element is different from the previous one, the alternating streak continues.
                # Any alternating subarray ending at i-1 can be extended by nums[i] to form an alternating subarray ending at i.
                # Plus, the single element subarray [nums[i]] is also alternating.
                # So, the number of alternating subarrays ending at i is 1 (for [nums[i]]) + (number of alternating subarrays ending at i-1).
                # The 'current_count_ending_here' from the previous iteration holds the number of alternating subarrays ending at i-1.
                current_count_ending_here = current_count_ending_here + 1

            # The total count is the sum of alternating subarrays ending at each index.
            total_count += current_count_ending_here

        return total_count