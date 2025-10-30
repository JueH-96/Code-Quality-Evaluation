class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected = set()
        operations = 0
        for i in range(len(nums) - 1, -1, -1):
            operations += 1
            collected.add(nums[i])
            target_set = set(range(1, k + 1))
            if collected.issuperset(target_set):
                return operations
        return operations