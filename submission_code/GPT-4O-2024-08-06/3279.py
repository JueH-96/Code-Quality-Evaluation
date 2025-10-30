class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Alice wins if the total number of flowers is odd
        # because she starts first and can always pick the last flower.
        # Therefore, we need to count pairs (x, y) such that (x + y) is odd.

        count = 0
        for x in range(1, n + 1):
            for y in range(1, m + 1):
                if (x + y) % 2 == 1:
                    count += 1
        return count