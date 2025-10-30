class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        remainder_counts = [0, 0, 0]
        for num in nums:
            remainder_counts[num % 3] += 1
        return min(remainder_counts[1], remainder_counts[2]) + remainder_counts[0]