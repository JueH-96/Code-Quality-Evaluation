class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        freq = [0] * 32
        
        for x in nums:
            for j in range(32):
                if (x >> j) & 1:
                    freq[j] += 1
        
        remove = [0] * (k + 1)
        initial_value = 0
        for j in range(32):
            if freq[j] > 0:
                initial_value += (1 << j)
                if freq[j] <= k:
                    remove[freq[j]] += (1 << j)
        
        current = initial_value
        res = 0
        for i in range(1, k + 1):
            res = (res + current * current) % mod
            if i < k:
                current -= remove[i]
        return res