class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        count = 0
        for x in range(1, n + 1):
            for y in range(1, m + 1):
                # Alice wins if total number of flowers is odd
                # Because she moves first and will pick the last flower
                if (x + y) % 2 == 1:
                    count += 1
        return count