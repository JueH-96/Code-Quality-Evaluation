class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        total_happiness = 0
        for _ in range(k):
            if not happiness:
                break
            total_happiness += happiness[0]
            for i in range(1, len(happiness)):
                happiness[i] = max(0, happiness[i] - 1)
            happiness.pop(0)
        return total_happiness