import collections
from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        """
        Calculates the distinct difference array for the given input array nums.

        The distinct difference array 'diff' is defined such that diff[i] is equal to 
        the number of distinct elements in the prefix nums[0, ..., i] minus 
        the number of distinct elements in the suffix nums[i + 1, ..., n - 1].

        Args:
            nums: A 0-indexed list of integers.

        Returns:
            A list of integers representing the distinct difference array.
            
        Constraints:
            1 <= n == nums.length <= 50
            1 <= nums[i] <= 50

        Examples:
            nums = [1,2,3,4,5] -> [-3,-1,1,3,5]
            nums = [3,2,3,4,2] -> [-2,-1,0,2,3]
            
        Approach:
            Use a single pass O(n) approach.
            Maintain a set for distinct elements in the prefix (`prefix_set`).
            Maintain a frequency map (`suffix_counts`) and a distinct count (`distinct_suffix_count`) 
            for the elements in the conceptual suffix starting from the current index `i`.
            In each iteration `i`:
            1. Update `prefix_set` by adding `nums[i]`. `prefix_distinct_count = len(prefix_set)`.
            2. Update `suffix_counts` by decrementing the count of `nums[i]`.
            3. If the count of `nums[i]` becomes 0 in `suffix_counts`, decrement `distinct_suffix_count`. 
               After this, `distinct_suffix_count` represents the number of distinct elements 
               in the actual suffix `nums[i+1...n-1]`.
            4. Calculate `diff[i] = prefix_distinct_count - distinct_suffix_count`.
        """
        n = len(nums)
        
        # Set to store distinct elements encountered in the prefix nums[0...i]
        prefix_set = set()
        
        # Frequency map for elements in the conceptual suffix starting from the current index i.
        # Initialized with counts for the entire array (suffix starting at index 0).
        suffix_counts = collections.Counter(nums)
        
        # Count of distinct elements in the conceptual suffix starting from the current index i.
        # Initialized with the number of distinct elements in the whole array.
        # This variable will be updated to hold the count for the suffix nums[i+1...n-1].
        distinct_suffix_count = len(suffix_counts)
        
        # Result array to store the differences
        diff = [0] * n
        
        # Iterate through the array, calculating diff[i] for each index i
        for i in range(n):
            current_num = nums[i]
            
            # --- Update Prefix State ---
            # Add the current element to the prefix set
            prefix_set.add(current_num)
            # Get the number of distinct elements in the prefix nums[0...i]
            prefix_distinct_count = len(prefix_set)
            
            # --- Update Suffix State ---
            # Adjust the suffix state to reflect the suffix nums[i+1...n-1]
            # by removing the contribution of nums[i].
            
            # Decrement the frequency of the current element in the suffix counts.
            # This signifies moving nums[i] from the suffix part to the prefix part.
            suffix_counts[current_num] -= 1
            
            # If the frequency becomes 0 after decrementing, it means this element 
            # is no longer present in the suffix nums[i+1...n-1].
            # Therefore, decrement the count of distinct elements for the suffix.
            if suffix_counts[current_num] == 0:
                distinct_suffix_count -= 1
                # After this update, `distinct_suffix_count` correctly holds the 
                # number of distinct elements in the suffix nums[i+1...n-1].
                
            # --- Calculate Difference ---
            # Calculate the difference for index i: 
            # (number of distinct elements in prefix nums[0...i]) - 
            # (number of distinct elements in suffix nums[i+1...n-1])
            diff[i] = prefix_distinct_count - distinct_suffix_count
            
        # Return the calculated distinct difference array
        return diff