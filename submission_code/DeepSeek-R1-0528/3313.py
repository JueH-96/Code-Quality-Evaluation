class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        NEG_INF = -10**18
        g_prev = [NEG_INF] * (k + 1)
        g_prev[0] = 0
        f_prev = [NEG_INF] * (k + 1)
        
        w = [(k - j) * (-1)**j for j in range(k)]
        
        for num in nums:
            new_f = [NEG_INF] * (k + 1)
            for j in range(1, k + 1):
                extend = f_prev[j] + w[j - 1] * num
                new_start = g_prev[j - 1] + w[j - 1] * num
                new_f[j] = max(extend, new_start)
            for j in range(1, k + 1):
                if new_f[j] > g_prev[j]:
                    g_prev[j] = new_f[j]
            f_prev = new_f
        
        return g_prev[k]