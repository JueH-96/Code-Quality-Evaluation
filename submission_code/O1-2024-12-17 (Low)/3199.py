class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        # We simply try all possible distributions for the 3 children.
        for x in range(limit + 1):
            for y in range(limit + 1):
                z = n - (x + y)
                # If z is within the allowable range, it's a valid distribution.
                if 0 <= z <= limit:
                    count += 1
        return count