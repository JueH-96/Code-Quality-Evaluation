class Solution:
    def circularGameLosers(self, n: int, k: int) -> list[int]:
        players = list(range(1, n+1))
        losers = []
        index = 0
        for i in range(1, n):
            index = (index + i*k) % len(players)
            losers.append(players.pop(index))
        return sorted(losers[:-1])