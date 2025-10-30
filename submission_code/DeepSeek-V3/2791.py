class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = set()
        current = 1
        step = 1
        while True:
            if current in received:
                break
            received.add(current)
            current = (current + step * k - 1) % n + 1
            step += 1
        losers = []
        for i in range(1, n+1):
            if i not in received:
                losers.append(i)
        return losers