class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        bit_count = [0] * 31  # bits from 0 to 30
        
        # Count the number of 1s in each bit position
        for num in nums:
            for i in range(31):
                if num & (1 << i):
                    bit_count[i] += 1
        
        result = 0
        # Generate k numbers with maximum possible values
        for _ in range(k):
            current = 0
            # Check each bit from highest to lowest
            for i in range(30, -1, -1):
                if bit_count[i] > 0:
                    current += (1 << i)
                    bit_count[i] -= 1
            # Add the square of the current number to the result
            result = (result + current * current) % MOD
        
        return result