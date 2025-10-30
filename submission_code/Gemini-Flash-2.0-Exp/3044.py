class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        needed = set(range(1, k + 1))
        count = 0
        for i in range(len(nums) - 1, -1, -1):
            count += 1
            if nums[i] in needed:
                needed.remove(nums[i])
            if not needed:
                return count
        return count