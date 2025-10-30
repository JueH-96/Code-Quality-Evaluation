class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        present = [False] * n
        for x in changeIndices:
            if 1 <= x <= n:
                present[x-1] = True
        if not all(present):
            return -1
        
        lo, hi = 1, m
        ans = -1
        
        def valid(T):
            last_occ = [-1] * n
            for i in range(T):
                idx = changeIndices[i] - 1
                last_occ[idx] = i
                
            if any(x == -1 for x in last_occ):
                return False
                
            pairs = [(last_occ[i], nums[i]) for i in range(n)]
            pairs.sort()
            
            for k in range(n):
                t, num_req = pairs[k]
                if t - k < num_req:
                    return False
            return True
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if valid(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
                
        return ans