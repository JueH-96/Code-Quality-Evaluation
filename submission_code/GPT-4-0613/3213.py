class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        indices = [i for i, x in enumerate(nums) if x == max_num]
        if len(indices) < k:
            return 0
        count = 0
        for i in range(len(indices) - k + 1):
            left = indices[i] - (indices[i - 1] if i > 0 else -1) - 1
            right = (indices[i + k] if i + k < len(indices) else len(nums)) - indices[i + k - 1] - 1
            count += (left + 1) * (right + 1)
        return count