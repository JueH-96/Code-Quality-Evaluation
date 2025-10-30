from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # dp_prev[i] will hold the best score for selecting k items
        # ending exactly at index i in b for the previous k
        dp_prev = [0] * n
        
        # Base case k=0: pick one element with weight a[0]
        for i in range(n):
            dp_prev[i] = a[0] * b[i]
        
        # For k = 1,2,3 compute dp_cur using dp_prev
        for k in range(1, 4):
            dp_cur = [0] * n
            # track the best dp_prev up to j < i
            best_prev = dp_prev[0]
            # We cannot pick i=0 for k>=1 because we need at least k picks before
            dp_cur[0] = float('-inf')
            for i in range(1, n):
                # update best prev up to index i-1
                best_prev = max(best_prev, dp_prev[i-1])
                # pick b[i] as the k-th element
                dp_cur[i] = best_prev + a[k] * b[i]
            dp_prev = dp_cur
        
        # answer is the best of dp_prev for k=3
        return max(dp_prev)

# Example usage:
# sol = Solution()
# print(sol.maxScore([3,2,5,6], [2,-6,4,-5,-3,2,-7]))  # 26
# print(sol.maxScore([-1,4,5,-2], [-5,-1,-3,-2,-4]))  # -1