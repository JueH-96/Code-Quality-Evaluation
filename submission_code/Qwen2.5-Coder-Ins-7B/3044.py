class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        target = set(range(1, k + 1))
        operations = 0
        collected = set()
        
        for i in range(len(nums) - 1, -1, -1):
            collected.add(nums[i])
            operations += 1
            if collected.issuperset(target):
                return operations