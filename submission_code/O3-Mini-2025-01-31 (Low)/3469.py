class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # The strategy is to simulate possible triangle heights (h)
        # and check if we can assign colors by alternating rows such that
        # one pattern (starting with red) or the other (starting with blue) is valid.
        #
        # If we start with red, then the red rows are the 1st, 3rd, 5th, ... rows.
        # The total balls needed in red rows equals the sum of odd numbers:
        #   sum_red = 1 + 3 + ... up to h (if h is odd, there are (h+1)//2 terms,
        #   and the sum equals ((h+1)//2)^2)
        # Similarly, the blue rows (2nd, 4th, 6th, ...) require:
        #   sum_blue = 2 + 4 + ... up to h (if h is even, there are (h//2) terms
        #   and the sum equals (h//2)*((h//2)+1))
        #
        # If the triangle is arranged alternating with the other color starting first,
        # then the roles of red and blue are simply swapped.
        #
        # We iterate over possible triangle heights until the total number of balls required
        # exceeds red+blue or no valid arrangement is possible.
        #
        # Since red, blue <= 100, the maximum possible h is modest.
        
        max_height = 0
        
        # The maximum h that might be possible is bounded by the total balls available.
        # h*(h+1)//2 <= red+blue, so h is at most around 20 when red+blue=200.
        # We iterate h from 1 up to 200 (a safe upper bound).
        for h in range(1, 201):
            # Calculate required balls for odd positions (first color when starting)
            odd_rows = (h + 1) // 2   # count of rows with the starting color
            even_rows = h // 2        # count of rows with the other color
            
            # Sum of balls used in odd-indexed rows (1, 3, 5, ...) = odd_rows^2
            req_odd = odd_rows * odd_rows
            # Sum of balls used in even-indexed rows (2, 4, 6, ...) = even_rows * (even_rows + 1)
            req_even = even_rows * (even_rows + 1)
            
            # Option 1: starting with red.
            valid_option1 = (red >= req_odd and blue >= req_even)
            # Option 2: starting with blue.
            valid_option2 = (blue >= req_odd and red >= req_even)
            
            if valid_option1 or valid_option2:
                max_height = h
            else:
                # If for current h no valid option is possible, further increasing h only increases requirements.
                # However, because sometimes a valid combination might appear for a larger h if the color totals are different,
                # we do not break early.
                pass
        
        return max_height

# Sample usage and test cases:
if __name__ == '__main__':
    sol = Solution()
    print(sol.maxHeightOfTriangle(2, 4))  # Expected output: 3
    print(sol.maxHeightOfTriangle(2, 1))  # Expected output: 2
    print(sol.maxHeightOfTriangle(1, 1))  # Expected output: 1
    print(sol.maxHeightOfTriangle(10, 1)) # Expected output: 2