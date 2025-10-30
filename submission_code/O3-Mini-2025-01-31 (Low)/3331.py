class Solution:
    def minOperations(self, nums: list[int], k: int) -> int:
        # Count how many elements are strictly less than k.
        # Since one operation removes one occurrence of the smallest element,
        # eventually, we need to remove all elements which are below k.
        return sum(1 for num in nums if num < k)