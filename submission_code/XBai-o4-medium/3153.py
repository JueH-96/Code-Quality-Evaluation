class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        cnt = [0] * 31  # Since 2^30 is slightly over 1e9
        
        for x in nums:
            for j in range(31):
                if (x >> j) & 1:
                    cnt[j] += 1
        
        res = 0
        for i in range(1, k + 1):
            current_val = 0
            for j in range(31):
                if cnt[j] >= i:
                    current_val += (1 << j)
            res += current_val * current_val
        
        return res % MOD