class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # For Alice to win, she must prevent Bob from picking the last flower, making the total flowers odd
        # Alice wins if total flowers are odd, since she starts first.
        
        # Initializing count of valid pairs
        count = 0
        
        # Checking all pairs of (x, y) within given ranges
        for x in range(1, n + 1):
            for y in range(1, m + 1):
                if (x + y) % 2 == 1:  # If total flowers (x + y) is odd, Alice wins
                    count += 1
        
        return count