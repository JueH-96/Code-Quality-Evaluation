class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of 'good' elements in the array.
        An element nums[i] is good if it is strictly greater than the elements
        at indices i - k and i + k (if those indices exist). If neither of these
        indices exists, nums[i] is still considered good.
        
        Args:
            nums: The input list of integers.
            k: The distance for checking neighboring elements.
            
        Returns:
            The sum of all good elements.
        """
        n = len(nums)
        total_sum = 0
        
        # Iterate through each element in the array
        for i in range(n):
            # Initialize conditions for being a good number
            is_strictly_greater_than_left_neighbor = True
            is_strictly_greater_than_right_neighbor = True
            
            # Check the left neighbor at index i - k
            left_index = i - k
            # If the left index is within bounds (>= 0), we must check the condition
            if left_index >= 0:
                if nums[i] <= nums[left_index]:
                    # If nums[i] is not strictly greater, the left condition is not met
                    is_strictly_greater_than_left_neighbor = False
            # If the left index is out of bounds (< 0), the condition for the left side
            # is considered met (as per the problem statement: if index doesn't exist,
            # that part is fine). The initial value True handles this case.

            # Check the right neighbor at index i + k
            right_index = i + k
            # If the right index is within bounds (< n), we must check the condition
            if right_index < n:
                if nums[i] <= nums[right_index]:
                    # If nums[i] is not strictly greater, the right condition is not met
                    is_strictly_greater_than_right_neighbor = False
            # If the right index is out of bounds (>= n), the condition for the right side
            # is considered met. The initial value True handles this case.
            
            # An element nums[i] is good if both conditions are met.
            # This covers cases where only one neighbor exists (only one check applies),
            # where both neighbors exist (both checks apply), and theoretically where
            # neither neighbor exists (neither check applies, both flags remain True).
            if is_strictly_greater_than_left_neighbor and is_strictly_greater_than_right_neighbor:
                total_sum += nums[i]
                
        return total_sum