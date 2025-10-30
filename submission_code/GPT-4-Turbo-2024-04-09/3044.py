class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        needed = set(range(1, k + 1))
        count = 0
        for num in reversed(nums):
            count += 1
            if num in needed:
                needed.remove(num)
            if not needed:
                break
        return count