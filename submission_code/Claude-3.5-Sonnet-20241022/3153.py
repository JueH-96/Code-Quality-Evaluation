class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Count bits at each position for all numbers
        bits = [0] * 32
        for num in nums:
            for j in range(32):
                if num & (1 << j):
                    bits[j] += 1
        
        # Construct k optimal numbers by greedily assigning bits
        result = 0
        for i in range(k):
            curr = 0
            # For each optimal number, take bits from most significant to least
            for j in range(31, -1, -1):
                if bits[j] > 0:
                    curr |= (1 << j)
                    bits[j] -= 1
            result = (result + curr * curr) % MOD
            
        return result