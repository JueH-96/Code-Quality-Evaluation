class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        # Count 1-bits at each position
        bit_count = [0] * 30
        for num in nums:
            for i in range(30):
                if num & (1 << i):
                    bit_count[i] += 1
        
        # Create k result numbers
        result = [0] * k
        
        # Assign bits from most significant to least significant
        for i in range(29, -1, -1):
            count = bit_count[i]
            # Sort to ensure we assign to the largest numbers
            result.sort(reverse=True)
            for j in range(min(k, count)):
                result[j] |= (1 << i)
        
        # Calculate sum of squares
        total = 0
        for num in result:
            total = (total + num * num) % MOD
        
        return total