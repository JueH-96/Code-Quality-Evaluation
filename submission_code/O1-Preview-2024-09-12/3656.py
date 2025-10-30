class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_K = (n + 2) // 3  # Maximum possible number of operations
        for K in range(max_K + 1):
            remaining_nums = nums[3 * K:]
            if len(set(remaining_nums)) == len(remaining_nums):
                # All elements are distinct
                return K
        return max_K