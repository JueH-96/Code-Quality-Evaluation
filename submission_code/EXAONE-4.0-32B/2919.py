class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        arr = sorted(usageLimits)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + arr[i - 1]
        
        idx_arr = [0] * (n + 1)
        ptr = 0
        for t_val in range(1, n + 1):
            while ptr < n and arr[ptr] < t_val:
                ptr += 1
            idx_arr[t_val] = ptr
        
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.check(mid, n, prefix_sum, idx_arr):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def check(self, k, n, prefix_sum, idx_arr):
        for t in range(1, k + 1):
            idx_t = idx_arr[t]
            S_t = prefix_sum[idx_t] + t * (n - idx_t)
            if S_t + t * (t - 1) // 2 < t * k:
                return False
        return True