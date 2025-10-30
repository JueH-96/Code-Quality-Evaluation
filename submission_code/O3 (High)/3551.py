from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # ------------------------------------------------------------------ #
        # 1. score[l][r]  -> XOR-score of the sub-array nums[l … r]          #
        #    Recurrence : score[l][r] = score[l][r-1] ^ score[l+1][r]        #
        # ------------------------------------------------------------------ #
        score = [[0] * n for _ in range(n)]
        for i in range(n):
            score[i][i] = nums[i]
        
        for length in range(2, n + 1):                 # length of the window
            for l in range(0, n - length + 1):
                r = l + length - 1
                score[l][r] = score[l][r - 1] ^ score[l + 1][r]

        # ------------------------------------------------------------------ #
        # 2. best[l][r]  -> maximum XOR-score of ANY sub-array completely    #
        #                  inside nums[l … r]                                #
        #    Recurrence : best[l][r] = max( score[l][r] ,                    #
        #                                   best[l+1][r] ,                   #
        #                                   best[l][r-1] )                   #
        #    (every sub-array of [l … r] is either:                          #
        #       a) the whole interval [l … r]                                #
        #       b) fully contained in [l+1 … r]                              #
        #       c) fully contained in [l … r-1])                             #
        # ------------------------------------------------------------------ #
        best = [[0] * n for _ in range(n)]
        for i in range(n):
            best[i][i] = nums[i]
        
        for length in range(2, n + 1):
            for l in range(0, n - length + 1):
                r = l + length - 1
                best[l][r] = max(score[l][r], best[l + 1][r], best[l][r - 1])

        # ------------------------------------------------------------------ #
        # 3. Answer every query in O(1) using the pre-computed best array.   #
        # ------------------------------------------------------------------ #
        return [best[l][r] for l, r in queries]