class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        operations = 0
        collected_elements = set()
        n = len(nums)
        required_elements = set(range(1, k + 1))

        for i in range(n - 1, -1, -1):
            operations += 1
            collected_elements.add(nums[i])
            if required_elements.issubset(collected_elements):
                return operations
        return operations