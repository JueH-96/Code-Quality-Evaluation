class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0] + 1
        
        cum = [0] * n
        for i in range(1, n):
            cum[i] = cum[i-1] + max(0, nums[i] - nums[i-1])
        
        mod = 10**9 + 7
        ans = 0
        
        for a in range(0, nums[0] + 1):
            con = [nums[i] - cum[i] - a for i in range(1, n)]
            F = [0] * 51
            if con[0] < 0:
                total_ways = 0
            else:
                bound0 = min(con[0], 50)
                for v in range(0, bound0 + 1):
                    F[v] = 1
                for j in range(1, n - 1):
                    if con[j] < 0:
                        F = [0] * 51
                        break
                    prefix_prev = [0] * 51
                    prefix_prev[0] = F[0]
                    for k in range(1, 51):
                        prefix_prev[k] = prefix_prev[k-1] + F[k]
                    F_new = [0] * 51
                    boundj = min(con[j], 50)
                    for v in range(0, boundj + 1):
                        F_new[v] = prefix_prev[v]
                    F = F_new
                total_ways = sum(F)
            ans = (ans + total_ways) % mod
        
        return ans