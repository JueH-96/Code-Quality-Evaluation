class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        original_min = min(nums)
        original_max = max(nums)
        adjusted_min = original_min - k
        adjusted_max = original_max + k
        possible_distinct = adjusted_max - adjusted_min + 1
        return min(len(nums), possible_distinct)