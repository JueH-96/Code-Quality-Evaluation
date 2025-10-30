class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Alice wins if she can force Bob to take the last flower
        # Alice starts, so she wins if total flowers (x + y) is odd
        # This is because if x + y is odd, the last move will be made by Bob (since Alice starts)
        
        # Count the number of valid (x, y) pairs where x + y is odd
        count = 0
        
        # Iterate over all possible values of x and y
        for x in range(1, n + 1):
            for y in range(1, m + 1):
                if (x + y) % 2 == 1:  # Check if x + y is odd
                    count += 1
        
        return count