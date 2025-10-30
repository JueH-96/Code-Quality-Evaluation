class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = [0] * (n + 1)
        current_friend = 1
        received[current_friend] = 1
        turn = 1
        while True:
            step = turn * k
            next_friend = (current_friend + step - 1) % n + 1
            if received[next_friend] == 1:
                break
            received[next_friend] = 1
            current_friend = next_friend
            turn += 1
        losers = []
        for i in range(1, n + 1):
            if received[i] == 0:
                losers.append(i)
        return losers