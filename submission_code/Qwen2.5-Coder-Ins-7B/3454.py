class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        return sum(abs(a - b) for a, b in zip(sorted(nums), sorted(target))) // 2