class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        n = len(nums)
        for i in range(n):
            for j in range(n):
                x = nums[i]
                y = nums[j]
                if abs(x - y) <= min(x, y):
                    current_xor = x ^ y
                    max_xor = max(max_xor, current_xor)
        return max_xor