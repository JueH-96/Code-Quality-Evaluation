class Solution:
    def canAliceWin(self, n: int) -> bool:
        # The game ends when the number of stones left is <= 0.
        # Alice starts with removing 10 stones, hence, if n <= 10, Alice could lose.
        # If n > 10, and is such that 1 <= n-10 <= 10, then Alice can win by making the remaining
        # stones <= 10 (if n is of the form 10+10, Alice can win), else, Alice will lose.
        # In essence, Alice wins if (n - 10) doesn't land exactly on n % 2, for that would mean
        # Bob can take n % 2 stones and leave Alice with nothing.
        
        # Alice wins if she can leave Bob with 11 or more stones.
        return n > 10 and ((n-10) % 2 == 0)