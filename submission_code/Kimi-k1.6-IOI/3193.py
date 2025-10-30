class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        max_xor = 0
        n = len(nums)
        for i in range(n):
            for j in range(n):
                x, y = nums[i], nums[j]
                if max(x, y) <= 2 * min(x, y):
                    current_xor = x ^ y
                    if current_xor > max_xor:
                        max_xor = current_xor
        return max_xor