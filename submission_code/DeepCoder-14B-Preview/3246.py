class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        evens = [x for x in nums if x % 2 == 0]
        return len(evens) >= 2