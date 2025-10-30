class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return -1

        # Count the number of zeros in the first three elements
        initial_zeros = nums[:3].count(0)

        # If the first three elements are all ones, we can make all elements ones
        if initial_zeros == 0:
            return 0

        # Initialize the minimum operations required
        min_operations = float('inf')

        # Iterate through the array and count the number of zeros in each window of three elements
        for i in range(n - 2):
            # Count the number of zeros in the current window of three elements
            current_zeros = nums[i:i+3].count(0)

            # If the current window has fewer zeros than the initial window, update the minimum operations
            if current_zeros < initial_zeros:
                initial_zeros = current_zeros
                min_operations = i + 1

        # If we found a valid window, return the minimum operations required
        if min_operations != float('inf'):
            return min_operations

        # If no valid window was found, return -1
        return -1