class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        # Iterate over candies given to first two children; third is determined
        for x in range(limit + 1):
            for y in range(limit + 1):
                z = n - x - y
                # Check if the third child gets a valid number of candies
                if 0 <= z <= limit:
                    count += 1
        return count