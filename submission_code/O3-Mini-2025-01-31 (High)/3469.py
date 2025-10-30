class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        import math
        
        # Total number of available balls.
        total = red + blue
        # Compute an upper bound h_max on the triangle height using total balls
        # because the sum of 1+2+...+h = h*(h+1)/2 should not exceed total.
        h_max = int((-1 + math.sqrt(1 + 8 * total)) // 2)
        
        max_height = 0
        
        # For each possible height from 1 to h_max, we check if a valid arrangement exists.
        # Since each row i must have exactly i balls and every row is monochromatic,
        # while adjacent rows must have different colors, the only two possibilities for the
        # color assignments are:
        #   Pattern 1: red for row 1, blue for row 2, red for row 3, ...
        #   Pattern 2: blue for row 1, red for row 2, blue for row 3, ...
        #
        # For Pattern 1:
        #   The odd rows (1, 3, 5, ...) use red and the sum of the first n odd numbers is n^2,
        #   where n = ceil(h/2)
        #   The even rows (2, 4, 6, ...) use blue and the sum of the first m even numbers is m*(m+1),
        #   where m = floor(h/2)
        #
        # For Pattern 2 the roles of red and blue are swapped.
        
        for h in range(1, h_max + 1):
            odd_count = (h + 1) // 2
            even_count = h // 2
            
            # For Pattern 1: red in odd rows, blue in even rows.
            red_needed_p1 = odd_count ** 2
            blue_needed_p1 = even_count * (even_count + 1)
            
            # For Pattern 2: blue in odd rows, red in even rows.
            blue_needed_p2 = odd_count ** 2
            red_needed_p2 = even_count * (even_count + 1)
            
            # Check if either pattern can be realized with the available red and blue balls.
            if ((red >= red_needed_p1 and blue >= blue_needed_p1) or
                (red >= red_needed_p2 and blue >= blue_needed_p2)):
                max_height = h
        
        return max_height

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxHeightOfTriangle(2, 4))   # Expected output: 3
    print(sol.maxHeightOfTriangle(2, 1))   # Expected output: 2
    print(sol.maxHeightOfTriangle(1, 1))   # Expected output: 1
    print(sol.maxHeightOfTriangle(10, 1))  # Expected output: 2