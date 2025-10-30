class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        times_received = [0] * (n + 1)

        current = 1  # 1st friend starts with the ball
        times_received[current] = 1

        i = 1
        while True:
            next_friend = (current - 1 + i * k) % n + 1
            times_received[next_friend] += 1
            if times_received[next_friend] == 2:
                break
            current = next_friend
            i += 1

        losers = [idx for idx in range(1, n + 1) if times_received[idx] == 0]
        return losers