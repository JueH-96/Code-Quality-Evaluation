class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        for i in range(len(nums)):
            for j in range(len(nums)):
                x = nums[i]
                y = nums[j]
                if abs(x - y) <= min(x, y):
                    max_xor = max(max_xor, x ^ y)
        return max_xor