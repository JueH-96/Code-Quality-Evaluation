class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = [False] * n
        current = 0
        received[0] = True
        i = 1
        while True:
            next_player = (current + i * k) % n
            if received[next_player]:
                break
            received[next_player] = True
            current = next_player
            i += 1
        losers = []
        for j in range(n):
            if not received[j]:
                losers.append(j + 1)
        return losers