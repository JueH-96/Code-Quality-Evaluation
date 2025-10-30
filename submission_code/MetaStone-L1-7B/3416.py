class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)
        if n == 0:
            return 0
        max_digits = len(str(nums[0]))
        for k in range(max_digits):
            counts = {}
            for num in nums:
                pow10 = 10 ** k
                digit = (num // pow10) % 10
                if digit in counts:
                    counts[digit] += 1
                else:
                    counts[digit] = 1
            same_pairs = 0
            for c in counts.values():
                same_pairs += c * (c - 1) // 2
            differing_pairs = (n * (n - 1) // 2) - same_pairs
            total_sum += differing_pairs
        return total_sum