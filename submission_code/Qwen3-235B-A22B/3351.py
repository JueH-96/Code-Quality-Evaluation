class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        total = 0
        for i in range(k):
            diff = happiness[i] - i
            if diff <= 0:
                break
            total += diff
        return total