class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        nums = []
        for r in range(m):
            for c in range(n):
                nums.append((grid[r][c], r, c))
        nums.sort(reverse=True)

        ans = 0
        rows = set()
        vals = set()

        for num, r, c in nums:
            if r not in rows and num not in vals:
                ans += num
                rows.add(r)
                vals.add(num)

        return ans