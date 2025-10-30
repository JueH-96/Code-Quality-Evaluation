from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # Prefix sums: pre[i] = sum of nums[:i]
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i+1] = pre[i] + nums[i]
        
        # Helper to get sum of nums[i..j], inclusive
        def seg_sum(i: int, j: int) -> int:
            return pre[j+1] - pre[i]
        
        # good[i][j] = can fully split segment nums[i..j] into singletons
        good = [[False] * n for _ in range(n)]
        
        # Segments of length 1 are trivially "splittable" (already single elements)
        for i in range(n):
            good[i][i] = True
        
        # Consider all lengths from 2 to n
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                # Try every possible split point k: [i..k] and [k+1..j]
                for k in range(i, j):
                    # Lengths of the two parts
                    len_left = k - i + 1
                    len_right = j - (k + 1) + 1
                    
                    # Check sum conditions for immediate split validity
                    left_ok = (len_left == 1) or (seg_sum(i, k) >= m)
                    right_ok = (len_right == 1) or (seg_sum(k+1, j) >= m)
                    if not (left_ok and right_ok):
                        continue
                    
                    # Check that each part can itself be fully split
                    if good[i][k] and good[k+1][j]:
                        good[i][j] = True
                        break  # found a valid split, no need to try more
        
        return good[0][n-1]