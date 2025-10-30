import math
import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())

    total_len = N * K

    # Precompute factorials up to total_len
    # Max total_len = 500 * 500 = 250000
    # math.factorial can handle large numbers.
    # Precomputation itself might be slow, but it's efficient enough in Python.
    fact = [1] * (total_len + 1)
    for i in range(1, total_len + 1):
        fact[i] = fact[i-1] * i

    # Function to calculate multinomial coefficient:
    # (n_total)! / (c_1! * c_2! * ... * c_m!)
    # where n_total = sum(c_i)
    # This function uses precomputed factorials.
    def calc_num_permutations(remaining_len, current_item_counts):
        res = fact[remaining_len]
        for c in current_item_counts:
            res //= fact[c]
        return res

    # Calculate total number of good sequences S
    # S = (NK)! / (K!)^N
    S = fact[total_len]
    for _ in range(N):
        S //= fact[K]

    # Calculate target rank (1-indexed)
    # floor((S+1)/2)
    target_rank = (S + 1) // 2

    # Counts of each number (1 to N), initially K for all
    # counts[0] is unused, counts[i] stores remaining count for number i.
    counts = [K] * (N + 1) 

    result_sequence = []

    # Iterate through each position of the sequence (from left to right)
    for current_pos in range(total_len):
        # remaining_len is total_len - current_pos
        remaining_len = total_len - current_pos

        # Try placing numbers from 1 to N in lexicographical order
        for num_val in range(1, N + 1):
            if counts[num_val] > 0:
                # Temporarily decrement count for num_val to simulate placing it
                counts[num_val] -= 1

                # Prepare counts for the calc_num_permutations function.
                # Only include counts of numbers that still have occurrences left.
                current_config_counts = []
                for i in range(1, N + 1):
                    if counts[i] > 0:
                        current_config_counts.append(counts[i])
                
                # If all counts are zero (meaning remaining_len - 1 is 0), 
                # there's only 1 permutation (empty sequence after current_pos).
                # This case is handled correctly by calc_num_permutations (fact[0] = 1).
                num_suffixes = calc_num_permutations(remaining_len - 1, current_config_counts)

                # Check if the target rank falls within this block of sequences
                # (sequences starting with the current prefix and num_val at current_pos)
                if target_rank > num_suffixes:
                    # Target rank is in a later block, so subtract this block's count
                    # and revert the temporary placement of num_val.
                    target_rank -= num_suffixes
                    counts[num_val] += 1  # Revert count for num_val
                else:
                    # Target rank is within this block, so num_val is the correct choice
                    result_sequence.append(num_val)
                    # The count for num_val is already decremented, keep it that way for the next position.
                    break # Move to the next position in the sequence

    # Print the resulting sequence
    sys.stdout.write(" ".join(map(str, result_sequence)) + "
")

solve()