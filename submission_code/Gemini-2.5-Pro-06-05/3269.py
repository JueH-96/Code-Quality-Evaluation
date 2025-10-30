from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)

        # Step 1: Transform the `nums` array into a "relationship" array.
        # This new array, `rel`, will represent the relationship (-1, 0, or 1)
        # between each pair of adjacent elements in `nums`.
        # A subarray `nums[i..i+m]` matches the `pattern` if and only if
        # the subarray `rel[i..i+m-1]` is identical to `pattern`.
        
        rel = []
        # The length of `rel` will be n-1.
        for i in range(n - 1):
            if nums[i + 1] > nums[i]:
                rel.append(1)
            elif nums[i + 1] == nums[i]:
                rel.append(0)
            else:  # nums[i + 1] < nums[i]
                rel.append(-1)
        
        # Step 2: Count the occurrences of `pattern` as a contiguous subarray in `rel`.
        # This is now a standard subarray/substring search problem.
        count = 0
        
        # The number of possible starting positions for a subarray of length `m`
        # in an array of length `n-1` is `(n-1) - m + 1 = n - m`.
        # So, we iterate `i` from 0 up to (but not including) `n - m`.
        for i in range(n - m):
            # Python's list slicing and comparison make this check concise and readable.
            # We check if the subarray of `rel` of length `m` starting at `i`
            # is identical to the `pattern`.
            if rel[i:i + m] == pattern:
                count += 1
        
        return count