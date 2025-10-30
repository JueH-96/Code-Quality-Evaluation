class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        maximumHeight.sort()
        total_sum = sum(maximumHeight)
        # Check if it's possible to assign heights
        for i in range(n):
            if maximumHeight[i] >= i + 1:
                continue
            else:
                return -1
        # Adjust the heights to ensure no two towers have the same height
        for i in range(n):
            maximumHeight[i] = n - i
            total_sum -= maximumHeight[i]
        return total_sum + sum(maximumHeight)