class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_val = 0
        for i in range(n):
            for j in range(i+1, n):
                if bin(nums[i] | nums[j]).count('1') <= k:
                    max_val = max(max_val, nums[i] ^ nums[j])
        return max_val