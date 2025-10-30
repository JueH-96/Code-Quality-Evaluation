class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        players = [{} for _ in range(n)]
        for x, y in pick:
            if y in players[x]:
                players[x][y] += 1
            else:
                players[x][y] = 1
        count = 0
        for i in range(n):
            max_c = 0
            if players[i]:
                max_c = max(players[i].values())
            if max_c > i:
                count += 1
        return count