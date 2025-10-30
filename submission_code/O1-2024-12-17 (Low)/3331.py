class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # The minimum number of operations is simply the count of elements
        # that are strictly less than k. Each of these must be removed
        # (one at a time, always removing the current smallest) until all
        # remaining elements are >= k.
        return sum(1 for num in nums if num < k)