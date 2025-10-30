class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        bit_count = [0] * 30  # Numbers up to 10^9 need at most 30 bits
        
        # Count how many 1s we have at each bit position across all numbers
        for num in nums:
            bit = 0
            while num:
                if num & 1:
                    bit_count[bit] += 1
                num >>= 1
                bit += 1
        
        # Construct k optimal numbers by greedily assigning bits
        # from most significant to least significant
        result_nums = [0] * k
        
        for i in range(29, -1, -1):  # From MSB to LSB
            count = bit_count[i]
            for j in range(min(k, count)):
                result_nums[j] |= (1 << i)
        
        # Calculate sum of squares
        sum_of_squares = 0
        for num in result_nums:
            sum_of_squares = (sum_of_squares + num * num) % MOD
        
        return sum_of_squares