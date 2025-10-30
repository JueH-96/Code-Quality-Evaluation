import bisect

class Solution:
    def minChanges(self, nums: list, k: int) -> int:
        n = len(nums)
        m = n // 2
        
        C = [0] * (k + 1)
        T_list = [[] for _ in range(k + 1)]
        freq_T = [0] * (k + 1)
        
        for i in range(m):
            a = nums[i]
            b = nums[n - 1 - i]
            d = abs(a - b)
            C[d] += 1
            T_i = max(a, k - a, b, k - b)
            T_list[d].append(T_i)
            freq_T[T_i] += 1
        
        D_arr = [0] * (k + 2)
        for X in range(1, k + 2):
            D_arr[X] = D_arr[X - 1] + (freq_T[X - 1] if X - 1 <= k else 0)
        
        for d in range(k + 1):
            if T_list[d]:
                T_list[d].sort()
        
        ans = float('inf')
        for X in range(0, k + 1):
            base = m - C[X]
            E = 0
            if T_list[X]:
                idx = bisect.bisect_left(T_list[X], X)
                E = idx
            extra = D_arr[X] - E
            total_cost = base + extra
            if total_cost < ans:
                ans = total_cost
        
        return ans