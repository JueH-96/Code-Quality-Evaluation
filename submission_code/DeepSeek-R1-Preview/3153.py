class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        bits = [0] * 31  # Since 1e9 is less than 2^30
        
        # Count the number of set bits for each position
        for num in nums:
            for i in range(31):
                if num & (1 << i):
                    bits[i] += 1
        
        top = [0] * k
        
        # Distribute the bits to maximize the sum of squares
        for i in range(30, -1, -1):
            m = min(bits[i], k)
            for j in range(m):
                top[j] += (1 << i)
            bits[i] -= m
        
        # Calculate the sum of squares
        result = 0
        for num in top:
            result = (result + num * num) % mod
        
        return result