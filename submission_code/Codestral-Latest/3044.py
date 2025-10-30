class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Create a set to keep track of the elements we need to collect
        target_set = set(range(1, k + 1))

        # Initialize the count of operations
        operations = 0

        # Iterate through the array from the end to the beginning
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] in target_set:
                target_set.remove(nums[i])
                operations += 1
            if not target_set:
                break

        return operations