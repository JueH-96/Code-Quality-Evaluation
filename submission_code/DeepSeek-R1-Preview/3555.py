class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        current_nums = list(nums)
        for _ in range(k):
            min_val = min(current_nums)
            idx = current_nums.index(min_val)
            current_nums[idx] = min_val * multiplier
        return current_nums