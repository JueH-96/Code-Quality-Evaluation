class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if any(x < k for x in nums):
            return -1
        if all(x == k for x in nums):
            return 0
        unique = sorted({x for x in nums if x > k})
        return len(unique)