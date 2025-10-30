class Solution:
    def minOperations(self, k: int) -> int:
        # Initialize the array with [1]
        nums = [1]
        # Initialize the sum of the array
        current_sum = 1
        # Initialize the number of operations
        operations = 0

        # While the current sum is less than k
        while current_sum < k:
            # If the current sum is less than half of k, duplicate the largest element
            if current_sum < k / 2:
                nums.append(nums[-1])
                current_sum += nums[-1]
                operations += 1
            else:
                # Otherwise, increase the largest element
                nums[-1] += 1
                current_sum += 1
                operations += 1

        return operations