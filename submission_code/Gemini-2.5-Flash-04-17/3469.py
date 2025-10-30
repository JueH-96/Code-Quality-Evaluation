class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        h = 0
        while True:
            h += 1
            
            # Calculate balls needed for current height h
            # Rows 1, 3, 5, ... (odd-indexed rows) form the first color group.
            # The number of rows in this group up to height h is ceil(h/2).
            # The balls needed are the sum of the first ceil(h/2) odd numbers: 1 + 3 + ...
            # The sum of the first k odd numbers is k^2.
            # Number of odd rows = (h + 1) // 2 (integer division handles ceil for positive h)
            num_odd_rows = (h + 1) // 2
            needed_odd_rows_sum = num_odd_rows * num_odd_rows

            # Rows 2, 4, 6, ... (even-indexed rows) form the second color group.
            # The number of rows in this group up to height h is floor(h/2).
            # The balls needed are the sum of the first floor(h/2) even numbers: 2 + 4 + ...
            # The sum of the first k even numbers is k * (k + 1).
            # Number of even rows = h // 2 (integer division handles floor for positive h)
            num_even_rows = h // 2
            needed_even_rows_sum = num_even_rows * (num_even_rows + 1)

            # Check possibility 1: Row 1 is red. Red balls fill odd rows, Blue balls fill even rows.
            can_start_red = (red >= needed_odd_rows_sum) and (blue >= needed_even_rows_sum)

            # Check possibility 2: Row 1 is blue. Blue balls fill odd rows, Red balls fill even rows.
            can_start_blue = (blue >= needed_odd_rows_sum) and (red >= needed_even_rows_sum)

            # If neither possibility is feasible for height h,
            # then the maximum achievable height is h-1.
            if not can_start_red and not can_start_blue:
                return h - 1