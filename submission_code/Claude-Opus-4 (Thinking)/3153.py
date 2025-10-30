class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        # Count bits at each position
        bit_count = [0] * 30  # 10^9 < 2^30
        
        for num in nums:
            for i in range(30):
                if num & (1 << i):
                    bit_count[i] += 1
        
        # Create k numbers
        result_nums = [0] * k
        
        # Distribute bits optimally
        for i in range(30):
            # Put bit i in the first min(bit_count[i], k) numbers
            for j in range(min(bit_count[i], k)):
                result_nums[j] |= (1 << i)
        
        # Calculate sum of squares
        result = 0
        for num in result_nums:
            result = (result + num * num) % MOD
        
        return result