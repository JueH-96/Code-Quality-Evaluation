class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # The total number of balls that can be used is red + blue.
        # The triangle has h rows and uses h*(h+1)/2 balls.
        # Also, each row is painted entirely in one color and adjacent rows must have different colors.
        # We have the freedom of choosing the starting color (red or blue).
        #
        # When starting with red:
        #   - The odd rows (1, 3, 5, …) are red. Their total ball count is the sum of the first m odd numbers (where m = ceil(h/2)). 
        #     This sum is m^2.
        #   - The even rows (2, 4, 6, …) are blue. Their total ball count is the sum of the first n even numbers (n = floor(h/2)).
        #     This sum is n*(n+1).
        #
        # Similarly, when starting with blue, the roles reverse.
        #
        # We can try all triangle heights h from 1 up to the maximum possible height such that
        # h*(h+1)/2 <= red + blue. For each h, we check if either starting arrangement is possible.
        
        max_h = 0
        total = red + blue
        # h*(h+1)//2 is the number of balls required. Given constraints are small, so we simply check all.
        h = 1
        while h * (h + 1) // 2 <= total:
            # When starting with red:
            red_rows = (h + 1) // 2  # number of rows that must be red (rows 1,3,5,...)
            blue_rows = h // 2       # number of rows that must be blue (rows 2,4,...)
            # Sum of first red_rows odd numbers = (red_rows)^2.
            red_needed1 = red_rows * red_rows
            # Sum of first blue_rows even numbers = blue_rows*(blue_rows + 1).
            blue_needed1 = blue_rows * (blue_rows + 1)
            
            # When starting with blue (the roles of red and blue are switched):
            blue_needed2 = red_rows * red_rows  # now blue covers the odd rows.
            red_needed2 = blue_rows * (blue_rows + 1)
            
            if (red_needed1 <= red and blue_needed1 <= blue) or (red_needed2 <= red and blue_needed2 <= blue):
                max_h = h
            h += 1
        return max_h

# Below are some test cases to validate the solution.
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.maxHeightOfTriangle(2, 4))  # Expected output: 3
    # Example 2:
    print(sol.maxHeightOfTriangle(2, 1))  # Expected output: 2
    # Example 3:
    print(sol.maxHeightOfTriangle(1, 1))  # Expected output: 1
    # Example 4:
    print(sol.maxHeightOfTriangle(10, 1))  # Expected output: 2