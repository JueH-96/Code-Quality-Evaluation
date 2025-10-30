from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # Each node stores a 2x2 matrix flattened as [M00, M01, M10, M11]
        # M[i][j] = max sum in segment if first taken state = i, last taken state = j
        INF = 10**18
        
        size = 1
        while size < n:
            size <<= 1
        seg = [[-INF]*4 for _ in range(2*size)]
        
        # build leaves
        for i in range(n):
            v = nums[i]
            node = size + i
            # M00 = 0, M11 = v, others invalid
            seg[node][0] = 0      # first=0, last=0
            seg[node][1] = -INF   # first=0, last=1
            seg[node][2] = -INF   # first=1, last=0
            seg[node][3] = v      # first=1, last=1
        # for unused leaves
        for i in range(n, size):
            node = size + i
            seg[node][0] = 0
            seg[node][1] = seg[node][2] = seg[node][3] = -INF
        
        def pull(k):
            # combine children 2k,2k+1 into k
            L = seg[2*k]
            R = seg[2*k+1]
            C = seg[k]
            # reset
            for idx in range(4):
                C[idx] = -INF
            # C[i][k] = max over L[i][j1] + R[j2][k] with not(j1==1 and j2==1)
            for i in (0,1):
                for j1 in (0,1):
                    lv = L[i*2 + j1]
                    if lv < -INF//2: continue
                    for j2 in (0,1):
                        if j1 == 1 and j2 == 1:
                            continue
                        for k2 in (0,1):
                            rv = R[j2*2 + k2]
                            if rv < -INF//2: continue
                            idx = i*2 + k2
                            s = lv + rv
                            if s > C[idx]:
                                C[idx] = s
        
        # build internal
        for k in range(size-1, 0, -1):
            pull(k)
        
        def update(pos, val):
            # set leaf at pos
            k = size + pos
            seg[k][0] = 0
            seg[k][1] = -INF
            seg[k][2] = -INF
            seg[k][3] = val
            k >>= 1
            while k:
                pull(k)
                k >>= 1
        
        def query_all():
            root = seg[1]
            # answer is max of all four entries
            return max(root)
        
        ans = 0
        for p, x in queries:
            update(p, x)
            res = query_all()
            # subsequence sum cannot be negative: empty subsequence allowed
            if res < 0:
                res = 0
            ans = (ans + res) % MOD
        return ans