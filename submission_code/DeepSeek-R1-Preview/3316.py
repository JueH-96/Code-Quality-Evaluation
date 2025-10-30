class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        ans = 0
        
        def compute_subsets(B, d):
            m = len(B)
            if m == 0:
                return [1] + [0] * (k)
            dp = [[0] * (m + 1) for _ in range(m)]
            for i in range(m):
                dp[i][1] = 1
            for s in range(2, m + 1):
                for i in range(m):
                    for j in range(i):
                        if B[i] - B[j] >= d:
                            dp[i][s] += dp[j][s - 1]
                            dp[i][s] %= MOD
            res = [0] * (m + 1)
            res[0] = 1
            for s in range(1, m + 1):
                total = 0
                for i in range(m):
                    total += dp[i][s]
                    total %= MOD
                res[s] = total
            return res
        
        for i in range(n):
            for j in range(i + 1, n):
                d = nums[j] - nums[i]
                left = []
                for x in range(i):
                    if nums[i] - nums[x] >= d:
                        left.append(nums[x])
                right = []
                for y in range(j + 1, n):
                    if nums[y] - nums[j] >= d:
                        right.append(nums[y])
                left_subsets = compute_subsets(left, d)
                right_subsets = compute_subsets(right, d)
                total_ways = 0
                max_l = len(left_subsets) - 1
                max_r = len(right_subsets) - 1
                for l in range(0, max_l + 1):
                    for r in range(0, max_r + 1):
                        if l + r + 2 == k:
                            total_ways = (total_ways + left_subsets[l] * right_subsets[r]) % MOD
                ans = (ans + d * total_ways) % MOD
        return ans % MOD