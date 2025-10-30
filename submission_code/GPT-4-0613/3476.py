class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        remainder = [0, 0, 0]
        for num in nums:
            remainder[num % 3] += 1
        return min(remainder[1], remainder[2]) + abs(remainder[1] - remainder[2]) // 3 * 2