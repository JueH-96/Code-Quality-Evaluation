class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        maximumHeight.sort(reverse=True)

        for i in range(n):
            if maximumHeight[i] < n - i:
                return -1

        assigned_heights = set()
        total_sum = 0

        for max_h in maximumHeight:
            h = max_h
            while h > 0 and h in assigned_heights:
                h -= 1

            if h > 0:
                assigned_heights.add(h)
                total_sum += h

        return total_sum