class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        total_candies = limit * 2
        if n > total_candies:
            return total_candies
        else:
            return n