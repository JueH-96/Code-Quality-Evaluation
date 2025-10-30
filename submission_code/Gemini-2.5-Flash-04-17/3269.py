from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        """
        Counts the number of subarrays in nums that match the given pattern.

        A subarray nums[i..j] of size m + 1 matches the pattern if the relationships
        between adjacent elements in the subarray match the pattern elements.
        Specifically, for k from 0 to m-1:
        nums[i + k + 1] > nums[i + k] if pattern[k] == 1
        nums[i + k + 1] == nums[i + k] if pattern[k] == 0
        nums[i + k + 1] < nums[i + k] if pattern[k] == -1

        Args:
            nums: A 0-indexed integer array of size n.
            pattern: A 0-indexed integer array of size m consisting of -1, 0, 1.

        Returns:
            The count of subarrays in nums that match the pattern.
        """
        n = len(nums)
        m = len(pattern)
        
        # Step 1: Build diff_pattern array
        # We can precompute the relationship between adjacent elements in nums.
        # Let diff_pattern[j] represent the relationship between nums[j+1] and nums[j].
        # diff_pattern[j] = 1 if nums[j+1] > nums[j]
        # diff_pattern[j] = 0 if nums[j+1] == nums[j]
        # diff_pattern[j] = -1 if nums[j+1] < nums[j]
        # This array will have size n-1, as there are n-1 pairs of adjacent elements.
        # The relationship between nums[j] and nums[j+1] is stored at index j in diff_pattern.
        
        diff_pattern = []
        # Iterate through nums from index 0 up to n-2 to consider each pair (nums[j], nums[j+1])
        for j in range(n - 1):
            if nums[j+1] > nums[j]:
                diff_pattern.append(1)
            elif nums[j+1] < nums[j]:
                diff_pattern.append(-1)
            else: # nums[j+1] == nums[j]
                diff_pattern.append(0)
        
        # The size of diff_pattern is N_dp = n - 1.
        # The size of pattern is M_p = m.

        # Step 2: Count occurrences of pattern (size m) as a subarray within diff_pattern (size n-1)
        # A subarray nums[i..i+m] of size m+1 starts at index i in nums and ends at index i+m.
        # This subarray involves the comparisons:
        # nums[i+1] vs nums[i] (corresponds to diff_pattern[i])
        # nums[i+2] vs nums[i+1] (corresponds to diff_pattern[i+1])
        # ...
        # nums[i+m] vs nums[i+m-1] (corresponds to diff_pattern[i+m-1])
        
        # For nums[i..i+m] to match the pattern[0..m-1], the sequence
        # diff_pattern[i], diff_pattern[i+1], ..., diff_pattern[i+m-1] must match pattern[0], ..., pattern[m-1].
        # This sequence in diff_pattern is the slice diff_pattern[i : i + m].
        
        # We need to find how many times the pattern array appears as a contiguous subarray
        # within the diff_pattern array.
        
        # The pattern array has size m. The diff_pattern array has size n-1.
        # A slice of diff_pattern of size m can start at index `i`.
        # The last possible starting index `i` in diff_pattern for a slice of length m is
        # when the slice ends at the last index of diff_pattern (which is n-2).
        # So, `i + m - 1 = n - 2`, which means `i = n - 2 - (m - 1) = n - m - 1`.
        # The possible starting indices `i` in diff_pattern range from 0 up to n - m - 1.
        # The number of such starting indices is (n - m - 1) - 0 + 1 = n - m.
        # We can iterate `i` from 0 to n - m - 1. The range is range(n - m).
        
        count = 0
        # The loop iterates through all possible start indices `i` for a matching slice
        # in the diff_pattern array.
        # The constraints 1 <= m < n guarantee that n - m >= 1, so the range is valid.
        for i in range(n - m):
            # Check if the slice diff_pattern starting at index i with length m
            # is equal to the pattern array.
            if diff_pattern[i : i + m] == pattern:
                count += 1
                
        return count