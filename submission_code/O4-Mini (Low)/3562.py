from bisect import bisect_right
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        # Augment with original indices
        arr = []
        for i, (l, r, w) in enumerate(intervals):
            arr.append((l, r, w, i))
        # Sort by start point
        arr.sort(key=lambda x: x[0])
        starts = [x[0] for x in arr]
        # Compute next non-overlapping index for each i
        nxt = [0]*n
        for i in range(n):
            r = arr[i][1]
            # We need l_j > r to be non-overlapping
            j = bisect_right(starts, r)
            nxt[i] = j
        
        # dp[i][k] = (maxWeight, lex-smallest sorted seq of original indices) 
        # using intervals from i..n-1, choosing k intervals
        # We'll build dp bottom-up
        dp = [ [ (0, []) for _ in range(5) ] for _ in range(n+1) ]
        # dp[n][*] are already (0,[])
        
        for i in range(n-1, -1, -1):
            for k in range(0, 5):
                # Option1: skip i
                best_w, best_seq = dp[i+1][k]
                # Option2: take i, if k>0
                if k > 0:
                    w_i = arr[i][2]
                    w2, seq2 = dp[nxt[i]][k-1]
                    cand_w = w_i + w2
                    cand_seq = seq2 + [arr[i][3]]
                    cand_seq.sort()
                    if cand_w > best_w or (cand_w == best_w and cand_seq < best_seq):
                        best_w, best_seq = cand_w, cand_seq
                dp[i][k] = (best_w, best_seq)
        
        # Among k=0..4 pick best
        res_w = -1
        res_seq = []
        for k in range(5):
            w, seq = dp[0][k]
            if w > res_w or (w == res_w and seq < res_seq):
                res_w = w
                res_seq = seq
        return res_seq