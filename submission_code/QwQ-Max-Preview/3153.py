class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        count = [0] * 31
        
        # Count the number of 1s in each bit position
        for num in nums:
            for bit in range(31):
                if num & (1 << bit):
                    count[bit] += 1
        
        elements = [0] * k
        
        # Distribute bits from highest to lowest
        for bit in reversed(range(31)):
            m = min(count[bit], k)
            val = 1 << bit
            for i in range(m):
                elements[i] += val
        
        # Calculate the sum of squares
        sum_squares = sum((x * x) % MOD for x in elements) % MOD
        return sum_squares