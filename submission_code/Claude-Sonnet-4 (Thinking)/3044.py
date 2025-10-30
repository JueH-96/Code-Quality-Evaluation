class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        target = set(range(1, k + 1))
        collected = set()
        
        for i in range(len(nums) - 1, -1, -1):
            collected.add(nums[i])
            if target.issubset(collected):
                return len(nums) - i
        
        return len(nums)