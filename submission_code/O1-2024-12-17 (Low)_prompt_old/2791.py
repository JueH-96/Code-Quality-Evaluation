class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = set([1])      # 1st friend starts with the ball
        current = 1             # current holder of the ball
        i = 1                   # turn index

        while True:
            # Compute the next friend to receive the ball
            next_friend = ((current - 1 + i * k) % n) + 1

            # If next friend has already received the ball, game ends
            if next_friend in received:
                break

            # Otherwise, record this friend and move on
            received.add(next_friend)
            current = next_friend
            i += 1

        # Losers are those who never received the ball
        return [friend for friend in range(1, n + 1) if friend not in received]