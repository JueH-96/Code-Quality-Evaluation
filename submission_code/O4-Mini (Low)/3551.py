from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # g[i][j] = XOR-score of subarray nums[i..j]
        g = [[0]*n for _ in range(n)]
        for i in range(n):
            g[i][i] = nums[i]
        # Fill DP for g using recurrence: g[i][j] = g[i][j-1] ^ g[i+1][j]
        for length in range(2, n+1):
            for i in range(n-length+1):
                j = i + length - 1
                g[i][j] = g[i][j-1] ^ g[i+1][j]
        # best[i][j] = maximum g[x][y] for i<=x<=y<=j
        best = [[0]*n for _ in range(n)]
        # For each end index j, we scan starts i from j down to 0,
        # maintain mx = max(g[i][j], g[i+1][j], ..., g[j][j])
        for j in range(n):
            mx = 0
            for i in range(j, -1, -1):
                # update maximum for subarrays ending at j starting >= i
                if g[i][j] > mx:
                    mx = g[i][j]
                # best[i][j] is either the best up to j-1, or mx
                if i == j:
                    best[i][j] = mx
                else:
                    # best[i][j-1] is defined since j-1 >= i
                    v = best[i][j-1]
                    best[i][j] = v if v >= mx else mx
        # Answer queries in O(1)
        ans = []
        for l, r in queries:
            ans.append(best[l][r])
        return ans

# Example usage:
# sol = Solution()
# print(sol.maximumSubarrayXor([2,8,4,32,16,1], [[0,2],[1,4],[0,5]]))  # [12,60,60]
# print(sol.maximumSubarrayXor([0,7,3,2,8,5,1], [[0,3],[1,5],[2,4],[2,6],[5,6]]))  # [7,14,11,14,5]