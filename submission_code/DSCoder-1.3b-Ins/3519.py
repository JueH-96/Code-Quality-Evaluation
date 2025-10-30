class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        count = [0]*n
        for x, y in pick:
            count[x] += 1
        return sum(1 for c in count if c >= i)