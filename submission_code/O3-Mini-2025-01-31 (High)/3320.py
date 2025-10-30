from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # We are given that in each operation we remove the first two elements.
        # The score of the operation is defined as the sum of those two elements.
        # Since we want every performed operation to have the same score,
        # the very first operation sets the target score.
        # Then we repeatedly check the (current) first two numbers in the list.
        # If their sum equals the target we perform the operation (i.e. remove them),
        # otherwise we must stop.
        
        # If there are fewer than two elements, no operation can be performed.
        if len(nums) < 2:
            return 0
        
        # Set the target score equal to the sum of the first two elements.
        target = nums[0] + nums[1]
        operations = 0
        
        # We simulate the process of repeatedly deleting the first two elements.
        # Because the operation always removes the first two elements,
        # the pairs taken correspond to the original list elements at positions:
        # (0,1), (2,3), (4,5), ... until there are fewer than 2 left.
        i = 0
        while i + 1 < len(nums):
            if nums[i] + nums[i + 1] == target:
                operations += 1
            else:
                # Once a pair does not match the target, we must stop.
                break
            # Move to the next pair (simulate deleting the first two elements).
            i += 2
        
        return operations

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxOperations([3, 2, 1, 4, 5]))  # Expected output: 2
    print(sol.maxOperations([3, 2, 6, 1, 4]))  # Expected output: 1