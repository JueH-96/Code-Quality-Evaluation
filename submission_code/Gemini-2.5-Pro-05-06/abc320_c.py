import math
import itertools

def main():
    M = int(input())
    S_reels = [input() for _ in range(3)] # S_reels[i] is the string for reel i

    min_total_time = float('inf')

    # Iterate over all possible target digits '0' through '9'
    for digit_char_code in range(ord('0'), ord('9') + 1):
        target_digit = chr(digit_char_code)
        
        # For each reel, find all indices k (0 to M-1) where S_reels[i][k] == target_digit
        # possible_indices_for_reels[i] will be a list of such k for reel i
        possible_indices_for_reels = []
        
        possible_for_this_digit = True
        for i in range(3): # For each of the 3 reels
            indices_for_current_reel = []
            for k in range(M): # For each position in the reel string
                if S_reels[i][k] == target_digit:
                    indices_for_current_reel.append(k)
            
            if not indices_for_current_reel: # If current reel has no target_digit
                possible_for_this_digit = False
                break 
            possible_indices_for_reels.append(indices_for_current_reel)
        
        if not possible_for_this_digit:
            continue # Try next digit, as this one is not on all reels

        # possible_indices_for_reels[j] contains list of valid indices for reel j.
        # Iterate over all combinations of (p0, p1, p2)
        # where p0 is an index for reel 0, p1 for reel 1, p2 for reel 2.
        # itertools.product generates Cartesian product: P_0 x P_1 x P_2.
        # Each element in product_of_indices will be a tuple (p0, p1, p2).
        for p_tuple in itertools.product(*possible_indices_for_reels):
            # p_tuple is (index_for_reel0, index_for_reel1, index_for_reel2)
            # e.g., p_tuple[0] is the chosen k for S_reels[0] so it shows target_digit.
            
            current_min_max_t_for_this_p_combo = float('inf')
            
            # Reels are indexed 0, 1, 2.
            # Try all 3! = 6 permutations of stopping reels.
            # A permutation (r0, r1, r2) means reel r0 is prioritized for earliest time slot,
            # then r1, then r2.
            reel_indices_list = [0, 1, 2]
            for perm_reels_order in itertools.permutations(reel_indices_list):
                # perm_reels_order is a tuple like (reel_idx_A, reel_idx_B, reel_idx_C)
                
                t_values_for_reels = {} # Stores actual stop time for each reel_idx.
                times_taken_set = set() # Stores actual stop times used, to ensure distinctness.

                for reel_idx in perm_reels_order:
                    # Base time for this reel_idx is p_tuple[reel_idx].
                    # This is the index k for S_reels[reel_idx][k] to show target_digit.
                    # The earliest time t such that t % M == k is k itself.
                    time_val = p_tuple[reel_idx] 
                    
                    # If this time_val is already taken by a previous reel in the permutation,
                    # add M to it until an unused time slot is found.
                    while time_val in times_taken_set:
                        time_val += M
                    
                    t_values_for_reels[reel_idx] = time_val
                    times_taken_set.add(time_val)
                
                # The overall time for this specific setup (p_tuple, perm_reels_order) 
                # is the max of assigned stop times.
                max_t_for_this_perm = max(t_values_for_reels.values())
                
                current_min_max_t_for_this_p_combo = min(current_min_max_t_for_this_p_combo, max_t_for_this_perm)
            
            min_total_time = min(min_total_time, current_min_max_t_for_this_p_combo)

    if min_total_time == float('inf'):
        print("-1")
    else:
        print(min_total_time)

if __name__ == '__main__':
    main()