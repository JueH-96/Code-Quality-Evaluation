class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        active = [False] * 301
        f = [[0] * 300 for _ in range(301)]
        suf_arr = [[0] * 300 for _ in range(301)]
        ans = 1
        
        for x in nums:
            best_here = 1
            for v in range(1, 301):
                if not active[v]:
                    continue
                d_val = abs(x - v)
                prev_best = suf_arr[v][d_val]
                candidate = prev_best + 1
                if candidate < 2:
                    candidate = 2
                if candidate > f[x][d_val]:
                    f[x][d_val] = candidate
                if candidate > best_here:
                    best_here = candidate
            
            suf_arr[x][299] = f[x][299]
            for d in range(298, -1, -1):
                suf_arr[x][d] = max(f[x][d], suf_arr[x][d+1])
            
            active[x] = True
            
            if best_here > ans:
                ans = best_here
        
        return ans