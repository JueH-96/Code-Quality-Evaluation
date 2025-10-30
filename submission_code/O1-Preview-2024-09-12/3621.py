class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if min(nums) < k:
            return -1
        unique_elements = sorted(set(nums), reverse=True)
        operations = len(unique_elements) - 1
        if k not in unique_elements:
            operations += 1
        return operations