class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Alice wins exactly when the total number of flowers x+y is odd,
        # because each move removes exactly one flower and the player
        # who makes the last move wins.  If x+y is odd, Alice (first player)
        # gets the last move.
        #
        # Count how many pairs (x,y) with 1<=x<=n, 1<=y<=m have x+y odd.
        # That's: (# odd x)*(# even y) + (# even x)*(# odd y).
        
        odd_x = (n + 1) // 2
        even_x = n // 2
        odd_y = (m + 1) // 2
        even_y = m // 2
        
        return odd_x * even_y + even_x * odd_y