from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2
        
        # count_d[d] = number of pairs with current absolute difference = d
        count_d = [0] * (k + 1)
        # freq_M[M] = number of pairs whose one‑change max reachable difference is M
        freq_M = [0] * (k + 1)
        
        # Process each pair
        for i in range(m):
            a = nums[i]
            b = nums[n - 1 - i]
            d = abs(a - b)
            count_d[d] += 1
            # M is the maximum difference achievable by changing just one element
            M = max(a, b, k - a, k - b)
            freq_M[M] += 1
        
        # Build prefix sums of freq_M for fast queries of A(X) = number of pairs with M < X
        pref = [0] * (k + 1)
        pref[0] = freq_M[0]
        for v in range(1, k + 1):
            pref[v] = pref[v - 1] + freq_M[v]
        
        # For each X in [0..k], cost(X) = m + A(X) - count_d[X]
        # where A(X) = sum_{M < X} freq_M[M] = pref[X-1] if X >= 1 else 0
        ans = float('inf')
        for X in range(k + 1):
            if X == 0:
                A = 0
            else:
                A = pref[X - 1]
            cost = m + A - count_d[X]
            if cost < ans:
                ans = cost
        
        return ans