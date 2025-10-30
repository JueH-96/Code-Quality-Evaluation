class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        coverage = [[] for _ in range(n)]
        for j, q in enumerate(queries):
            l, r, val = q
            for i in range(l, r + 1):
                coverage[i].append((j, val))
        
        low, high = 0, m
        ans = -1
        while low <= high:
            mid = (low + high) // 2
            if self.check_feasible(mid, nums, coverage):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans

    def check_feasible(self, k, nums, coverage):
        n = len(nums)
        for i in range(n):
            total = nums[i]
            if total == 0:
                continue
            vals = [val for (j, val) in coverage[i] if j < k]
            if not vals:
                return False
            dp = [False] * (total + 1)
            dp[0] = True
            for v in vals:
                if dp[total]:
                    break
                set_total = False
                for x in range(total, -1, -1):
                    if dp[x] and x <= total - v:
                        if x + v <= total:
                            dp[x + v] = True
                            if x + v == total:
                                set_total = True
                                break
                if set_total or dp[total]:
                    break
            if not dp[total]:
                return False
        return True