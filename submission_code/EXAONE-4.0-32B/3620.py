class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        last = -10**18
        distinct = 0
        for x in nums:
            candidate = max(last + 1, x - k)
            if candidate <= x + k:
                distinct += 1
                last = candidate
        return distinct