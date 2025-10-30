class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        # Try all possible candies for child 1
        for child1 in range(min(n + 1, limit + 1)):
            # Try all possible candies for child 2
            for child2 in range(min(n - child1 + 1, limit + 1)):
                # Child 3 gets the remaining candies
                child3 = n - child1 - child2
                # Check if child3's candies are within valid range
                if 0 <= child3 <= limit:
                    count += 1
        return count