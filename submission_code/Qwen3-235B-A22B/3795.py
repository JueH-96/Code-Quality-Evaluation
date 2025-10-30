import bisect
from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        
        # Precompute for each element i the list of (val, j) pairs
        self.lists = [[] for _ in range(n)]
        self.list_j = [[] for _ in range(n)]
        self.prefix_sums = [[] for _ in range(n)]
        
        for j in range(m):
            l, r, val = queries[j]
            for i in range(l, r + 1):
                self.lists[i].append((val, j))
        
        for i in range(n):
            # Build list_j[i] and prefix_sums[i]
            self.prefix_sums[i].append(0)
            self.list_j[i] = []
            for (val, j) in self.lists[i]:
                self.prefix_sums[i].append(self.prefix_sums[i][-1] + val)
                self.list_j[i].append(j)
        
        low = 0
        high = m
        answer = -1
        while low <= high:
            mid = (low + high) // 2
            if self.is_possible(nums, mid):
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        return answer if answer != -1 else -1
    
    def is_possible(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if k == 0:
            return all(x == 0 for x in nums)
        
        for i in range(n):
            current_val = nums[i]
            j_list = self.list_j[i]
            t = bisect.bisect_left(j_list, k)
            sum_vals = self.prefix_sums[i][t]
            
            if sum_vals < current_val:
                return False
            
            vals = [v for (v, j) in self.lists[i][:t]]
            target = current_val
            if target == 0:
                continue
            
            # Subset sum check
            dp = [False] * (target + 1)
            dp[0] = True
            for v in vals:
                if v > target:
                    continue
                for j in range(target, v - 1, -1):
                    if dp[j - v]:
                        if not dp[j]:
                            dp[j] = True
            if not dp[target]:
                return False
        
        return True