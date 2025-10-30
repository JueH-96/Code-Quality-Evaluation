class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        # Iterate over possible candies for the first child
        for x in range(min(n, limit) + 1):
            # Iterate over possible candies for the second child
            for y in range(min(n - x, limit) + 1):
                # Calculate candies for the third child
                z = n - x - y
                # Check if the third child gets no more than the limit
                if 0 <= z <= limit:
                    count += 1
        return count