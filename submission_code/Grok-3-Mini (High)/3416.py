class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        N = len(nums)
        str_nums = [str(num) for num in nums]
        D = len(str_nums[0])
        total_sum = 0
        for idx in range(D):
            freq = [0] * 10
            for s in str_nums:
                digit = int(s[idx])
                freq[digit] += 1
            sum_freq_sq = sum(count ** 2 for count in freq)
            diff_pairs = (N * N - sum_freq_sq) // 2
            total_sum += diff_pairs
        return total_sum