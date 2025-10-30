class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        count = 0
        for num in nums:
            if num % 2 == 1:
                count += 1
        return count