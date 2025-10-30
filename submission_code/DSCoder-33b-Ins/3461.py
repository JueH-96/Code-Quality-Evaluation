class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        left, right, height = [0]*n, [n]*n, [0]*n
        ans = float('inf')
        for i in range(m):
            l, r = m, -1
            for j in range(n):
                if grid[i][j]:
                    height[j] += 1
                    left[j] = max(left[j], l)
                else:
                    height[j] = 0
                    left[j] = 0
                    l = j + 1
            for j in range(n-1, -1, -1):
                if grid[i][j]:
                    right[j] = min(right[j], r)
                    ans = min(ans, (right[j]-left[j])*height[j])
                else:
                    right[j] = n
                    r = j
        return ans if ans < float('inf') else 0