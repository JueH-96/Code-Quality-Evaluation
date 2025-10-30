class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        max_bit = 60
        freq = [0] * (max_bit + 1)
        
        for num in nums:
            for j in range(max_bit + 1):
                if num & (1 << j):
                    freq[j] += 1
        
        max_exp = 2 * max_bit
        power2 = [1] * (max_exp + 1)
        for i in range(1, max_exp + 1):
            power2[i] = (power2[i - 1] * 2) % mod
        
        ans = 0
        for j in range(max_bit + 1):
            cnt = min(freq[j], k)
            exp_sq = 2 * j
            term = power2[exp_sq]
            ans = (ans + cnt * term) % mod
        
        for j in range(max_bit + 1):
            for l in range(j + 1, max_bit + 1):
                cnt = min(freq[j], freq[l], k)
                exp_sum = j + l
                term = power2[exp_sum]
                ans = (ans + 2 * cnt * term) % mod
        
        return ans