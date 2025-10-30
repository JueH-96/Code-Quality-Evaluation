class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[2], reverse=True)
        houses = [0] * n
        total_gold = 0
        for start, end, gold in offers:
            for i in range(start, end + 1):
                if houses[i] == 0:
                    houses[i] = 1
                    total_gold += gold
                    break
        return total_gold