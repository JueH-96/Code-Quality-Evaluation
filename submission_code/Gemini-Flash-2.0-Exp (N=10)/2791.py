class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = [False] * n
        current = 0
        received[current] = True
        step = 1
        while True:
            current = (current + step * k) % n
            if received[current]:
                break
            received[current] = True
            step += 1
        losers = []
        for i in range(n):
            if not received[i]:
                losers.append(i + 1)
        return losers