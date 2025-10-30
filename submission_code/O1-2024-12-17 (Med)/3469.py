class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Helper functions for summing the first n odd/even numbers of a triangle row pattern
        def sum_of_odds(n: int) -> int:
            # 1 + 3 + 5 + ... + (2n-1) = n^2
            return n * n
        
        def sum_of_evens(n: int) -> int:
            # 2 + 4 + 6 + ... + 2n = 2*(1 + 2 + ... + n) = n*(n+1)
            return n * (n + 1)
        
        max_height = 0
        
        # We'll check all possible heights from 1 to a safe upper bound.
        # Given red, blue <= 100, we won't exceed height 19 or 20 (sum(1..20) = 210).
        for k in range(1, 21):
            # Number of odd rows in k rows
            odd_rows = (k + 1) // 2
            # Number of even rows in k rows
            even_rows = k // 2
            
            # Case 1: row 1 is red => odd rows are red, even rows are blue
            red_needed_1 = sum_of_odds(odd_rows)
            blue_needed_1 = sum_of_evens(even_rows)
            if red_needed_1 <= red and blue_needed_1 <= blue:
                max_height = k
            
            # Case 2: row 1 is blue => odd rows are blue, even rows are red
            blue_needed_2 = sum_of_odds(odd_rows)
            red_needed_2 = sum_of_evens(even_rows)
            if red_needed_2 <= red and blue_needed_2 <= blue:
                max_height = k
        
        return max_height