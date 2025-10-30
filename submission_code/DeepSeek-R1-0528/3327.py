import math

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        total_ones = sum(nums)
        ones_positions = [i for i, num in enumerate(nums) if num == 1]
        
        if k == 0:
            return 0
        
        ans = float('inf')
        for a in range(n):
            r0 = 1 if nums[a] == 1 else 0
            if k - r0 == 0:
                ans = min(ans, 0)
                continue
            
            available_initial_ones = total_ones - r0
            L = max(0, k - r0 - maxChanges)
            R = min(available_initial_ones, k - r0)
            if L > R:
                continue
            
            vals = []
            max_dist = min(a, n - 1 - a, 1000)
            for d in range(1, max_dist + 1):
                positions = []
                if a - d >= 0:
                    positions.append(a - d)
                if a + d < n:
                    positions.append(a + d)
                for pos in positions:
                    if nums[pos] == 1:
                        vals.append(d - 2)
            
            if a - 0 >= 0 and a - 0 < n and a != a:
                pass
            vals.sort()
            m = len(vals)
            if m < L:
                continue
            prefix = [0] * (m + 1)
            for i in range(1, m + 1):
                prefix[i] = prefix[i - 1] + vals[i - 1]
            
            current_min = float('inf')
            for t in range(L, min(m, R) + 1):
                current_min = min(current_min, prefix[t])
            if current_min == float('inf'):
                continue
            total_cost = 2 * (k - r0) + current_min
            ans = min(ans, total_cost)
        
        return ans if ans != float('inf') else 2 * k