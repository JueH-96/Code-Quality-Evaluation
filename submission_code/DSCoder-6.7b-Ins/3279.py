class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Initialize the count of possible pairs
        count = 0

        # Iterate over the range of possible flowers
        for x in range(1, n+1):
            for y in range(1, m+1):
                # If Alice can win the game, increment the count
                if (x-y)%2 == 1:
                    count += 1

        # Return the count of possible pairs
        return count