class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        bit_counts = [0] * 31
        for x in nums:
            for i in range(31):
                if (x >> i) & 1:
                    bit_counts[i] += 1
        n = len(nums)
        result_nums = [0] * n
        for i in range(31):
            count = bit_counts[i]
            for j in range(count):
                result_nums[j] |= (1 << i)
        
        ans = 0
        mod = 10**9 + 7
        for i in range(k):
            square = (result_nums[i] * result_nums[i]) % mod
            ans = (ans + square) % mod
            
        return ans