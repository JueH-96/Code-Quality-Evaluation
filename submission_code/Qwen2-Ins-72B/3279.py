class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        count = 0
        for x in range(1, n+1):
            for y in range(1, m+1):
                # Alice wins if the total number of flowers is odd
                if (x + y) % 2 != 0:
                    count += 1
        return count