class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        count = 0
        nums.sort()
        for num in nums:
            if num >= k:
                break
            count += 1
        return count