class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        maximumHeight.sort(reverse=True)
        s = set()
        total_sum = 0
        for h in maximumHeight:
            i = h
            while i > 0:
                if i not in s:
                    s.add(i)
                    total_sum += i
                    break
                i -= 1
            if i == 0:
                return -1
        return total_sum