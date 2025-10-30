# Import List from typing for type hinting
from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        """
        Counts the number of subarrays in nums that match the given pattern.

        A subarray nums[i..j] of size m + 1 matches the pattern if for each k from 0 to m-1:
        - nums[i + k + 1] > nums[i + k] if pattern[k] == 1.
        - nums[i + k + 1] == nums[i + k] if pattern[k] == 0.
        - nums[i + k + 1] < nums[i + k] if pattern[k] == -1.

        Args:
            nums: A list of integers, the main array.
            pattern: A list of integers (-1, 0, 1) representing the pattern to match.

        Returns:
            The count of subarrays in nums that match the pattern.
        """
        
        n = len(nums)  # Length of the input array nums
        m = len(pattern) # Length of the pattern array

        # The problem asks for subarrays of size m + 1.
        # The pattern constraints relate m pairs of adjacent elements within such a subarray.
        
        # We can transform the problem into finding occurrences of the 'pattern' list
        # within a derived sequence that represents the relationships between adjacent elements in 'nums'.

        # Step 1: Create the 'diff_pattern' array.
        # This array will store the relationship (-1, 0, or 1) between adjacent elements of 'nums'.
        # diff_pattern[i] will store the relationship between nums[i+1] and nums[i].
        diff_pattern = []
        # Iterate through nums from the first element up to the second-to-last element.
        # The loop runs n-1 times, generating a diff_pattern of length n-1.
        for i in range(n - 1): 
            # Compare nums[i+1] with nums[i]
            if nums[i+1] > nums[i]:
                diff_pattern.append(1)  # Represents an increase
            elif nums[i+1] == nums[i]:
                diff_pattern.append(0)  # Represents equality
            else: # nums[i+1] < nums[i]
                diff_pattern.append(-1) # Represents a decrease
        
        # The length of the derived sequence 'diff_pattern'.
        diff_n = len(diff_pattern) # This is equal to n - 1.
        
        # Edge case: If the pattern length 'm' is greater than the length of 'diff_pattern',
        # it's impossible for the pattern to exist as a sublist.
        if m > diff_n:
            return 0

        # Step 2: Count the occurrences of 'pattern' as a contiguous sublist within 'diff_pattern'.
        # We can use a simple sliding window approach (naive pattern matching).
        count = 0
        
        # We need to check all possible starting positions 'j' for a sublist of length 'm'
        # within 'diff_pattern'.
        # A sublist starting at index 'j' spans indices from j to j + m - 1.
        # The last possible starting index 'j' is 'diff_n - m'.
        # The Python range function needs the stop value to be one greater than the last desired index.
        # So, range(diff_n - m + 1) iterates 'j' from 0 to 'diff_n - m'.
        for j in range(diff_n - m + 1):
            # Extract the sublist of 'diff_pattern' starting at index 'j' with length 'm'.
            # Python's list slicing `[start:end]` creates a new list containing elements
            # from index `start` up to (but not including) `end`.
            # `diff_pattern[j : j + m]` correctly extracts the sublist of length m.
            sub_list = diff_pattern[j : j + m]
            
            # Compare the extracted sublist `sub_list` with the target `pattern`.
            # Python's list equality `==` performs an element-wise comparison.
            if sub_list == pattern:
                # If the sublist matches the pattern, increment the count.
                count += 1
                
        # Return the total count of matching subarrays found.
        return count