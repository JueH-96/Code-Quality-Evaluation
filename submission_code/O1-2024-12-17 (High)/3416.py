class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert all numbers to strings (they all have the same length of digits)
        str_nums = [str(num) for num in nums]
        n = len(nums)
        k = len(str_nums[0])  # number of digits per integer
        
        # Frequency array: freq[digit_position][digit] = count of how many
        # numbers have 'digit' at position 'digit_position'
        freq = [[0]*10 for _ in range(k)]
        
        # Fill the frequency array
        for s in str_nums:
            for i, ch in enumerate(s):
                freq[i][int(ch)] += 1
        
        # Calculate the sum of digit differences
        # For each digit-position, the number of differing pairs is:
        # (n^2 - sum over d of freq[i][d]^2) / 2
        result = 0
        for i in range(k):
            sum_sq = sum(count*count for count in freq[i])
            result += (n*n - sum_sq) // 2
        
        return result