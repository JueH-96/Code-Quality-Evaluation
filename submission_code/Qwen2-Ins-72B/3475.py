class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ones = sum(nums)
        if ones == 0:
            return 0
        if ones == n:
            return -1 if 0 in nums else 0
        if ones == 1:
            return -1
        ans = float('inf')
        ones_positions = [i for i, x in enumerate(nums) if x == 1]
        for i in range(1, len(ones_positions)):
            ans = min(ans, (ones_positions[i] - ones_positions[i-1]) // 3)
        return ans * 3 + (ones % 3)