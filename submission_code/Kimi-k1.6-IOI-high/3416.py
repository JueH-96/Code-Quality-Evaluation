class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        str_nums = list(map(str, nums))
        m = len(str_nums[0])  # All numbers have the same length
        total_sum = 0
        
        for i in range(m):
            freq = [0] * 10  # Using a list for digits 0-9
            for s in str_nums:
                digit = int(s[i])
                freq[digit] += 1
            same_pairs = 0
            for count in freq:
                same_pairs += count * (count - 1) // 2
            total_pairs = n * (n - 1) // 2
            diff_pairs = total_pairs - same_pairs
            total_sum += diff_pairs
        
        return total_sum