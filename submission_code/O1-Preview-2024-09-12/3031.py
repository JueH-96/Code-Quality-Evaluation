class Solution:
    def constructProductMatrix(self, grid):
        n = len(grid)
        m = len(grid[0])
        nums = []
        for row in grid:
            nums.extend(row)
        N = len(nums)
        prefix_products = [1] * (N + 1)
        suffix_products = [1] * (N + 1)

        mod = 12345

        for i in range(1, N + 1):
            prefix_products[i] = (prefix_products[i - 1] * nums[i - 1]) % mod

        for i in range(N - 1, -1, -1):
            suffix_products[i] = (suffix_products[i + 1] * nums[i]) % mod

        result = []
        index = 0
        for i in range(n):
            row_result = []
            for j in range(m):
                p = (prefix_products[index] * suffix_products[index + 1]) % mod
                row_result.append(p)
                index += 1
            result.append(row_result)
        return result