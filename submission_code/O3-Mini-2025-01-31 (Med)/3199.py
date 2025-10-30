class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        # Iterate over all possible distributions for 3 children (x, y, z)
        for x in range(limit + 1):
            for y in range(limit + 1):
                for z in range(limit + 1):
                    if x + y + z == n:
                        count += 1
        return count