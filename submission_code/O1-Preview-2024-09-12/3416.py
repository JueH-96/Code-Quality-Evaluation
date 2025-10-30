class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        D = len(str(nums[0]))
        n = len(nums)
        total_diff = 0
        nums_str = [str(num) for num in nums]
        for pos in range(D):
            counts = [0]*10
            for num_str in nums_str:
                digit = int(num_str[pos])
                counts[digit] += 1
            total_pairs = n * (n - 1) // 2
            same_digit_pairs = sum(count * (count - 1) // 2 for count in counts)
            diff_at_pos = total_pairs - same_digit_pairs
            total_diff += diff_at_pos
        return total_diff