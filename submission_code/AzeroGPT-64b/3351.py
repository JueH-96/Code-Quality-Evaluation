class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        res, dec = 0, 0
        for i in range(k):
            res += max(0, happiness[i]-dec)
            dec += 1
        return res