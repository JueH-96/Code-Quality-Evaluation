import math

class Solver:
    def __init__(self):
        self.memo_f = {}

    def get_f(self, r_i, r_j):
        # Normalize i,j for memoization key using 4x2 periodicity
        # Python's % behavior: e.g. -1 % 4 = 3, -1 % 2 = 1. This is suitable.
        i_norm = r_i % 4
        j_norm = r_j % 2
        
        if (i_norm, j_norm) in self.memo_f:
            return self.memo_f[(i_norm, j_norm)]

        # (r_i + r_j) % 2 != 0 means sum is odd. Python: (-1)%2 = 1 (odd), (-2)%2 = 0 (even).
        if (r_i + r_j) % 2 != 0: 
            result = 1
        else: # Sum is even
            # Python's // is floor division.
            # (r_i + floor(r_j/2) + floor((r_i+r_j)/2)) % 2
            color_val_sum = r_i + (r_j // 2) + ((r_i + r_j) // 2)
            
            if color_val_sum % 2 == 0: # Parity 0 means Black
                result = 2
            else: # Parity 1 means White
                result = 0
        
        self.memo_f[(i_norm, j_norm)] = result
        return result

    def solve_problem(self):
        A, B, C, D = map(int, input().split())

        W = C - A # Width of the rectangle in terms of unit cells
        H = D - B # Height of the rectangle in terms of unit cells
        
        ans = 0
        
        # Precompute/populate memo_f for one 4x2 period block.
        # Also calculate sum_4x2_block, which should be 8.
        sum_4x2_block = 0
        for i_offset in range(4): # Covers i relative to 0, e.g., 0,1,2,3
            for j_offset in range(2): # Covers j relative to 0, e.g., 0,1
                sum_4x2_block += self.get_f(i_offset, j_offset)
        
        # Part 1: Full 4x2 blocks contribution
        num_full_blocks_W = W // 4
        num_full_blocks_H = H // 2
        ans += num_full_blocks_W * num_full_blocks_H * sum_4x2_block
        
        # Part 2: Contribution from the right strip of remaining columns
        # Width: W % 4, Height: num_full_blocks_H * 2
        if num_full_blocks_H > 0:
            # Start i for this strip (first column after full blocks)
            start_i_strip_W = A + num_full_blocks_W * 4 
            for col_idx in range(W % 4):
                current_i = start_i_strip_W + col_idx
                # Sum for this column `current_i` over `num_full_blocks_H` pairs of rows starting at `B`.
                # Due to j-periodicity, value for (current_i, B) + (current_i, B+1) is constant for B+2k.
                ans += num_full_blocks_H * (self.get_f(current_i, B) + self.get_f(current_i, B + 1))

        # Part 3: Contribution from the bottom strip of remaining rows
        # Width: num_full_blocks_W * 4, Height: H % 2 (either 0 or 1 row)
        if H % 2 == 1: # If there's one remaining row
            # j-coordinate of this remaining row
            current_j = B + num_full_blocks_H * 2
            if num_full_blocks_W > 0:
                # Sum over one 4-cell segment of this row (from A to A+3 for this current_j)
                sum_row_segment_4 = 0
                for i_offset in range(4):
                    sum_row_segment_4 += self.get_f(A + i_offset, current_j)
                ans += num_full_blocks_W * sum_row_segment_4
            
        # Part 4: Contribution from the corner block
        # Width: W % 4, Height: H % 2
        if H % 2 == 1: # If there's one remaining row for the corner
            current_j = B + num_full_blocks_H * 2
            start_i_corner = A + num_full_blocks_W * 4
            for col_idx in range(W % 4): # Iterate remaining W % 4 columns
                current_i = start_i_corner + col_idx
                ans += self.get_f(current_i, current_j)
                
        print(ans)

solver_instance = Solver()
solver_instance.solve_problem()