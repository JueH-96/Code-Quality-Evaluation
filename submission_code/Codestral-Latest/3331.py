class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Count the number of elements in nums that are less than k
        count = sum(1 for num in nums if num < k)
        return count