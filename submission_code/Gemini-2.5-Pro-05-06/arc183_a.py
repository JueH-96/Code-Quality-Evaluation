import math
import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())

    # Calculate initial S_current = (N*K)! / (K!)^N
    # This is the total number of good sequences S.
    
    val_fact_K = math.factorial(K)
    val_fact_NK = math.factorial(N * K)
    
    denominator = pow(val_fact_K, N)
    
    current_S_val = val_fact_NK // denominator
    
    # Target rank is floor((S+1)/2)-th, which is ceil(S/2)-th
    target_rank = (current_S_val + 1) // 2
    
    # counts[i] is the number of times (i+1) needs to be placed
    counts = [K] * N 
    ans = []
    
    remaining_len = N * K
    
    # Loop for each position in the sequence
    for _idx_pos in range(N * K):
        # Loop for each possible number (1 to N) to place at the current position
        for val_choice_idx in range(N): # Try to place number val_choice_idx + 1
            if counts[val_choice_idx] == 0: # If this number has no remaining occurrences
                continue
            
            # Calculate how many sequences can be formed if we pick (val_choice_idx + 1) here.
            # This is S_current * counts[val_choice_idx] / remaining_len.
            # Multiplication must be done first to maintain precision with integer arithmetic.
            # The division S_current * counts[val_choice_idx] / remaining_len is exact.
            num_sequences_with_this_choice = current_S_val * counts[val_choice_idx] // remaining_len
            
            if target_rank <= num_sequences_with_this_choice:
                # This is the number we pick for the current position.
                ans.append(val_choice_idx + 1)
                
                # Update state for the next position:
                current_S_val = num_sequences_with_this_choice 
                counts[val_choice_idx] -= 1
                remaining_len -= 1
                break # Move to determine the next element in the sequence
            else:
                # This number (val_choice_idx + 1) is not picked.
                # Subtract its count from target_rank and try the next possible number.
                target_rank -= num_sequences_with_this_choice
                # current_S_val remains the same for this position, as it represents the total
                # ways to complete the suffix from the state _before_ trying val_choice_idx.
                
    sys.stdout.write(" ".join(map(str, ans)) + "
")

if __name__ == '__main__':
    solve()