class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = [False] * (n + 1)
        current = 1
        received[current] = True
        turn = 1
        while True:
            next_friend = (current + turn * k) % n
            if next_friend == 0:
                next_friend = n
            if received[next_friend]:
                break
            received[next_friend] = True
            current = next_friend
            turn += 1
        losers = []
        for i in range(1, n + 1):
            if not received[i]:
                losers.append(i)
        return losers