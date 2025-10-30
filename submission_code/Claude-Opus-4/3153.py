class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        # Count the number of 1-bits at each position
        bit_count = [0] * 30  # 30 bits is enough for 10^9
        
        for num in nums:
            for i in range(30):
                if num & (1 << i):
                    bit_count[i] += 1
        
        # Create k numbers by distributing the bits optimally
        result_nums = [0] * k
        
        # For each bit position, distribute the available 1-bits to the first available numbers
        for i in range(30):
            count = bit_count[i]
            for j in range(min(count, k)):
                result_nums[j] |= (1 << i)
        
        # Calculate the sum of squares
        result = 0
        for num in result_nums:
            result = (result + num * num) % MOD
        
        return result