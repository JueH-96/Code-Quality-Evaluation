class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        satisfied = [False] * n
        count_sat = 0
        dp = []
        for i in range(n):
            if nums[i] == 0:
                satisfied[i] = True
                count_sat += 1
                d = [False] * (nums[i] + 1)
                d[0] = True
                dp.append(d)
            else:
                d = [False] * (nums[i] + 1)
                d[0] = True
                dp.append(d)
        
        if count_sat == n:
            return 0
        
        for k, (l, r, val) in enumerate(queries):
            for i in range(l, r + 1):
                if satisfied[i]:
                    continue
                for s in range(nums[i], val - 1, -1):
                    if dp[i][s - val]:
                        dp[i][s] = True
                if dp[i][nums[i]]:
                    satisfied[i] = True
                    count_sat += 1
                    if count_sat == n:
                        return k + 1
        return -1