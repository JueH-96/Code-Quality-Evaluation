from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Helper: given a horizontal line at y = L, compute (total area above - total area below)
        # For each square with bottom = y and side = l, its square spans [y, y+l]. 
        # - If L <= y: whole square is above, diff = l*l.
        # - If L >= y+l: whole square is below, diff = -l*l.
        # - If y < L < y+l: then the square is split:
        #       area_above = (y+l - L)*l   (vertical segment length times width l)
        #       area_below = (L - y)*l
        #    so diff = (y+l - L)*l - (L - y)*l = l*( (y+l - L) - (L - y) ) = l*(2*y + l - 2*L)
        #    which can be written as 2*l*(y + l/2 - L).
        def f(L: float) -> float:
            total = 0.0
            for y, _, l in squares:
                top = y + l
                if L <= y:
                    total += l * l
                elif L >= top:
                    total -= l * l
                else:
                    total += 2 * l * (y + l/2 - L)
            return total
        
        # Determine search boundaries: 
        # The valid L must lie between the minimum bottom and maximum top among all squares.
        min_y = min(y for y, _, _ in squares)
        max_top = max(y + l for y, _, l in squares)
        
        # At L = min_y, every square is either completely or partially above.
        # At L = max_top, every square is completely below.
        # So f(min_y) should be >= 0 while f(max_top) should be <= 0.
        lo = min_y
        hi = max_top
        
        # Binary search for L such that f(L) == 0 with required precision.
        # Function f(L) is continuous and strictly decreasing.
        # We'll use a tolerance of 1e-7 to ensure our answer within 1e-5 accuracy.
        eps = 1e-7
        
        # Binary search loop
        while hi - lo > eps:
            mid = (lo + hi) / 2
            if f(mid) > 0:
                # If the difference is positive, then the line is too low
                # so we need to move the line upward (increase L) so that terms become smaller.
                lo = mid
            else:
                hi = mid
        
        return (lo + hi) / 2


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    squares1 = [[0, 0, 1], [2, 2, 1]]
    result1 = sol.separateSquares(squares1)
    print(f"Example 1: {result1:.5f}")  # Expected: 1.00000
    
    # Example 2
    squares2 = [[0, 0, 2], [1, 1, 1]]
    result2 = sol.separateSquares(squares2)
    print(f"Example 2: {result2:.5f}")  # Expected: about 1.16667