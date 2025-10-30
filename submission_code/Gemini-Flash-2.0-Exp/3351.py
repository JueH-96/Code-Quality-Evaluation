class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        ans = 0
        for i in range(k):
            ans += happiness[i]
            for j in range(k, len(happiness)):
                happiness[j] = max(0, happiness[j] - 1)
        return ans