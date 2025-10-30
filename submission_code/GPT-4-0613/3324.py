class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        return len(set(nums)) >= len(nums) / 2