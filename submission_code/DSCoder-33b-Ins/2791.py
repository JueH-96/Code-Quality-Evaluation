class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        friends = [0] * n
        i, step = 0, 1
        while friends[i] < 2:
            friends[i] += 1
            i = (i + step * k) % n
            step += 1
        return [i + 1 for i in range(n) if friends[i] == 0]