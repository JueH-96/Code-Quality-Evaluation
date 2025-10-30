class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Initialize count of possible pairs
        count = 0
        
        # Iterate over all possible values of x and y
        for x in range(1, n + 1):
            for y in range(1, m + 1):
                # Check if Alice can win
                if (x + y) % 2 == 1:
                    count += 1
        
        # Return the count of possible pairs
        return count