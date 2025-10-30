class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        def digit_difference(x, y):
            return sum(a != b for a, b in zip(str(x), str(y)))

        total_sum = 0
        n = len(nums)

        for i in range(n):
            for j in range(i + 1, n):
                total_sum += digit_difference(nums[i], nums[j])

        return total_sum