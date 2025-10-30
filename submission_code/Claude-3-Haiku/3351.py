class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        n = len(happiness)
        happiness.sort(reverse=True)
        total_happiness = 0
        for i in range(k):
            total_happiness += happiness[i]
            for j in range(i+1, n):
                happiness[j] = max(0, happiness[j]-1)
        return total_happiness