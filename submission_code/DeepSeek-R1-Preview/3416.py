class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        str_nums = [str(num) for num in nums]
        n = len(str_nums)
        d = len(str_nums[0]) if n > 0 else 0
        total = 0

        for i in range(d):
            current_digits = [s[i] for s in str_nums]
            freq = {}
            for digit in current_digits:
                if digit in freq:
                    freq[digit] += 1
                else:
                    freq[digit] = 1
            same_pairs = 0
            for count in freq.values():
                same_pairs += count * (count - 1) // 2
            total_pairs = n * (n - 1) // 2
            total += (total_pairs - same_pairs)
        return total