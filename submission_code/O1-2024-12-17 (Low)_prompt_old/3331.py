class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # The key insight is that each operation removes exactly one smallest element,
        # and we only need to remove all elements that are strictly less than k.
        # The number of such elements is therefore the answer.
        return sum(1 for x in nums if x < k)