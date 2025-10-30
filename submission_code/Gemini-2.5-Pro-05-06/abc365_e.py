import sys

def solve():
    N = int(sys.stdin.readline())
    A_list = list(map(int, sys.stdin.readline().split()))

    # prefix_xor_sums[k] will store A[0]^...^A[k-1] for 0-indexed A_list
    # prefix_xor_sums[0] = 0 (XOR sum of empty prefix)
    # prefix_xor_sums[1] = A_list[0]
    # prefix_xor_sums[i] = A_list[0] ^ ... ^ A_list[i-1]
    prefix_xor_sums = [0] * (N + 1) 
    
    current_xor_sum = 0
    for i in range(N):
        current_xor_sum ^= A_list[i]
        prefix_xor_sums[i+1] = current_xor_sum
    
    total_sum_val = 0
    
    # Max A_i is 10^8. $10^8 < 2^{27}$.
    # So bits 0 to 26 are potentially non-zero. range(27) for b.
    # Using range(28) covers bits 0 to 27 for a small safety margin.
    # Max prefix_xor_sum will also be < 2^27.
    for b in range(28): 
        
        current_bit_contribution_count = 0 # This is S_b for current bit b
        
        # count_zeros: number of P_x (prefix_xor_sums[x_idx])
        # such that bit b of P_x is 0.
        # count_ones: similar, but bit b of P_x is 1.
        # These counts are for P_x where x_idx ranges from 0 up to (current_y_idx - 2).
        count_zeros = 0
        count_ones = 0
        
        # Loop y_idx from 0 to N. prefix_xor_sums[y_idx] is a potential P_y.
        # P_y must have index y_idx >= 2, because P_x has index x_idx <= y_idx - 2, and min x_idx is 0.
        for y_idx in range(N + 1):
            
            if y_idx >= 2:
                # The prefix sum P_x = prefix_xor_sums[x_idx] that is now available for pairing
                # (i.e., its index x_idx = y_idx - 2 is valid) is added to counts.
                # These counts are for P_0, ..., P_{y_idx-2}.
                x_val_to_add_to_counts = prefix_xor_sums[y_idx - 2]
                
                if ((x_val_to_add_to_counts >> b) & 1) == 0: # bit b of P_x is 0
                    count_zeros += 1
                else: # bit b of P_x is 1
                    count_ones += 1

                # Current P_y is prefix_xor_sums[y_idx].
                # We need to check its b-th bit.
                y_val = prefix_xor_sums[y_idx]
                bit_b_of_y = (y_val >> b) & 1
                
                if bit_b_of_y == 1: # bit b of P_y is 1
                    # We need P_x where bit b is 0, for (P_y ^ P_x)'s bit b to be 1.
                    current_bit_contribution_count += count_zeros
                else: # bit b of P_y is 0
                    # We need P_x where bit b is 1, for (P_y ^ P_x)'s bit b to be 1.
                    current_bit_contribution_count += count_ones
            
        total_sum_val += (current_bit_contribution_count * (1 << b))
        
    sys.stdout.write(str(total_sum_val) + "
")

solve()