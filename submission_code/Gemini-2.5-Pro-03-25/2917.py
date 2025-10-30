from typing import List # Import List type hint

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        """
        Counts the number of pairs (i, j) such that 0 <= i < j < n and nums[i] + nums[j] < target.

        Args:
            nums: A 0-indexed list of integers.
            target: An integer target value.

        Returns:
            The total number of pairs (i, j) satisfying the conditions 0 <= i < j < n 
            and nums[i] + nums[j] < target.
        
        Constraints:
            1 <= nums.length == n <= 50
            -50 <= nums[i], target <= 50

        Examples:
            Input: nums = [-1,1,2,3,1], target = 2
            Output: 3
            Explanation: The pairs are (0, 1), (0, 2), (0, 4) because:
            - nums[0] + nums[1] = -1 + 1 = 0 < 2
            - nums[0] + nums[2] = -1 + 2 = 1 < 2
            - nums[0] + nums[4] = -1 + 1 = 0 < 2

            Input: nums = [-6,2,5,-2,-7,-1,3], target = -2
            Output: 10

        Approach:
            The problem asks us to find the number of pairs of indices (i, j) such that i comes before j (i < j)
            and the sum of the elements at these indices is strictly less than the target value.
            A straightforward approach is to iterate through all possible pairs of indices (i, j) that satisfy 
            0 <= i < j < n. This can be achieved using nested loops.
            The outer loop iterates from i = 0 to n-1.
            The inner loop iterates from j = i + 1 to n-1. This ensures that j is always greater than i.
            Inside the inner loop, we check if the sum nums[i] + nums[j] is less than the target.
            If the condition nums[i] + nums[j] < target is true, we increment a counter.
            After iterating through all valid pairs, the counter will hold the total number of pairs satisfying the condition.

        Time Complexity: O(n^2), where n is the length of the input list `nums`. We use two nested loops, 
                         each potentially running up to n times. Given the constraint n <= 50, n^2 is at most 2500, 
                         which is well within typical time limits.
        Space Complexity: O(1), as we only use a few variables (n, count, i, j) to store state, requiring constant extra space.
        """
        
        n = len(nums) # Get the length of the input list
        count = 0     # Initialize a counter for the number of valid pairs
        
        # Iterate through the first element of the potential pair using index i
        for i in range(n):
            # Iterate through the second element of the potential pair using index j
            # Start j from i + 1 to ensure that j > i, satisfying the condition 0 <= i < j < n
            # This also prevents counting the same pair twice (e.g., (a, b) and (b, a)) and pairing an element with itself (i == j).
            for j in range(i + 1, n):
                # Check if the sum of the elements at indices i and j is strictly less than the target value
                if nums[i] + nums[j] < target:
                    # If the condition is met, increment the counter
                    count += 1
                    
        # After checking all valid pairs, return the final count
        return count