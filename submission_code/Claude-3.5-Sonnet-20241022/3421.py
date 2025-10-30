class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        n = len(hours)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                total = hours[i] + hours[j]
                if total % 24 == 0:
                    count += 1
        return count