class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        memo = {}
        def dp(low: int, high: int, left: int, right: int) -> int:
            if low == high and left == right:
                return 0
            if (low, high, left, right) in memo:
                return memo[(low, high, left, right)]
            ans = float('inf')
            # Horizontal cuts
            for i in range(low, high):
                cost = horizontalCut[i] + dp(low, i, left, right) + dp(i + 1, high, left, right)
                if cost < ans:
                    ans = cost
            # Vertical cuts
            for j in range(left, right):
                cost = verticalCut[j] + dp(low, high, left, j) + dp(low, high, j + 1, right)
                if cost < ans:
                    ans = cost
            memo[(low, high, left, right)] = ans
            return ans
        return dp(0, m - 1, 0, n - 1)