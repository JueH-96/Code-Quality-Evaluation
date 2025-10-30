class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        count = 0
        for x in range(1, n+1):
            for y in range(1, m+1):
                if x % 2 == 1 or y % 2 == 1:
                    count += 1
        return count