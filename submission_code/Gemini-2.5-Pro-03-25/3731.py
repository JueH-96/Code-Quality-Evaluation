import math # Not needed, but included in standard LeetCode environment
from typing import List # Required for type hinting

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        """
        Calculates the total sum of elements from specific subarrays defined for each index.

        For each index i (0 <= i < n), a subarray nums[start ... i] is defined, 
        where start = max(0, i - nums[i]). This function computes the sum of all 
        elements across all n such subarrays by directly iterating and summing.

        Args:
            nums: A list of integers. The input array.

        Returns:
            An integer representing the total sum calculated according to the rules.
        
        Example 1:
        Input: nums = [2,3,1]
        Output: 11
        Explanation:
        i=0: start=max(0, 0-2)=0. Subarray nums[0...0]=[2]. Sum=2.
        i=1: start=max(0, 1-3)=0. Subarray nums[0...1]=[2,3]. Sum=5.
        i=2: start=max(0, 2-1)=1. Subarray nums[1...2]=[3,1]. Sum=4.
        Total Sum = 2 + 5 + 4 = 11.

        Example 2:
        Input: nums = [3,1,1,2]
        Output: 13
        Explanation:
        i=0: start=max(0, 0-3)=0. Subarray nums[0...0]=[3]. Sum=3.
        i=1: start=max(0, 1-1)=0. Subarray nums[0...1]=[3,1]. Sum=4.
        i=2: start=max(0, 2-1)=1. Subarray nums[1...2]=[1,1]. Sum=2.
        i=3: start=max(0, 3-2)=1. Subarray nums[1...3]=[1,1,2]. Sum=4.
        Total Sum = 3 + 4 + 2 + 4 = 13.

        Constraints:
        1 <= n == nums.length <= 100
        1 <= nums[i] <= 1000
        
        Time Complexity: O(n^2), where n is the length of nums. The outer loop runs n times,
                         and the inner loop can run up to n times in the worst case 
                         (when start_index is 0). Given n <= 100, O(n^2) is acceptable (100*100 = 10,000 operations).
        Space Complexity: O(1) additional space (excluding the input array itself). We only use a few variables 
                          to store the sums and indices.
        """
        # Get the length of the input array
        n = len(nums)
        # Initialize the total sum accumulator to 0. This will store the final result.
        total_sum = 0
        
        # Iterate through each index 'i' of the array 'nums', from 0 up to n-1.
        for i in range(n):
            # Calculate the start index for the subarray defined for the current index 'i'.
            # The start index is the maximum of 0 and (i - value at index i), as per the problem definition.
            # We use max(0, ...) to ensure the start index is always non-negative and within the array bounds.
            start_index = max(0, i - nums[i])
            
            # Initialize the sum for the current subarray (which spans from start_index to i, inclusive).
            current_subarray_sum = 0
            
            # Iterate through the indices 'j' from start_index up to and including 'i'.
            # This inner loop calculates the sum of elements within the defined subarray for the current index 'i'.
            # Python's range(start, end) excludes 'end', so we use range(start_index, i + 1) to include index 'i'.
            for j in range(start_index, i + 1):
                # Add the element at index 'j' to the sum of the current subarray.
                current_subarray_sum += nums[j]
                
            # After calculating the sum of the subarray for index 'i', add this sum 
            # to the overall total sum.
            total_sum += current_subarray_sum
            
        # After iterating through all indices 'i' and summing their respective subarray sums,
        # return the final accumulated total sum.
        return total_sum