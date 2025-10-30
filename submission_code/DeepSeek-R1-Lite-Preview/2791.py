class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = set()
        received.add(1)
        current_holder = 1
        i = 1
        while True:
            next_holder = (current_holder + i * k - 1) % n + 1
            if next_holder in received:
                break
            received.add(next_holder)
            current_holder = next_holder
            i += 1
        losers = [friend for friend in range(1, n+1) if friend not in received]
        return losers