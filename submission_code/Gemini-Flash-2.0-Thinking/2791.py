class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = {1}
        current_friend = 1
        turn = 1

        while True:
            steps = turn * k
            next_friend = (current_friend + steps - 1) % n + 1

            if next_friend in received:
                break

            received.add(next_friend)
            current_friend = next_friend
            turn += 1

        losers = []
        for i in range(1, n + 1):
            if i not in received:
                losers.append(i)

        return sorted(losers)