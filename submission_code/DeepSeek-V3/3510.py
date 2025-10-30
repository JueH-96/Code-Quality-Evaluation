class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        total = 0
        prev = 0
        for i in range(len(maximumHeight)):
            current = min(maximumHeight[i], prev + 1)
            if current <= 0:
                return -1
            total += current
            prev = current
        return total