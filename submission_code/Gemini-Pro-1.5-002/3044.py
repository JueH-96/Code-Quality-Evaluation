class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected = set()
        count = 0
        for i in range(len(nums) - 1, -1, -1):
            count += 1
            collected.add(nums[i])
            if len(collected.intersection(set(range(1, k + 1)))) == k:
                return count
        return count