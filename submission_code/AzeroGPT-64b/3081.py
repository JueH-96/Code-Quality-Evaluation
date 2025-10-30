class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        c = Counter(nums)
        first, second = sorted(c.values(), reverse=True)[:2]
        if first > second + 1:
            first -= 1
        return (first + first % 2) // 2