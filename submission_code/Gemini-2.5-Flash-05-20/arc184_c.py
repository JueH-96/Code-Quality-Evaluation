import sys

# Set recursion limit (default is too low for some inputs)
sys.setrecursionlimit(2000)

memo = {}
CHUNK_SIZE = 60 # A_N <= 10^18 < 2^60, so 60 bits are enough to cover A_k values.

def solve(offsets, current_power):
    # Sort and convert to tuple for memoization key
    t_offsets = tuple(sorted(offsets))

    if (t_offsets, current_power) in memo:
        return memo[(t_offsets, current_power)]

    # Base case: No more bits to consider. No further contributions.
    if current_power == 0:
        return (0, 0) # (max_score_if_q_v2_is_even, max_score_if_q_v2_is_odd)

    # Determine the number of bits to process in this chunk
    chunk_p = min(current_power, CHUNK_SIZE)

    max_even_q_score = 0
    max_odd_q_score = 0

    # Determine candidate `r` (low bits of `i`) values.
    # These are `0` and `(-A_k) % (1 << chunk_p)` for each `A_k`
    # because these are the `r` values that might cause `r + A_k` to be a multiple of `1 << chunk_p`.
    candidate_r_values = set()
    candidate_r_values.add(0) 
    for a_val in offsets:
        candidate_r_values.add((-a_val) % (1 << chunk_p))
    
    for r in candidate_r_values:
        score_for_this_r_chunk = 0
        next_offsets = []
        
        for a_val in offsets:
            val = r + a_val
            
            # Use `val.trailing_zeros()` for Python integers
            # If val is 0, trailing_zeros is undefined/infinite.
            # However, i >= 1, A_k >= 0, so i + A_k >= 1. val will always be >= 1.
            v2_val = val.trailing_zeros()
            
            if v2_val < chunk_p: # v_2 of val is determined by its lower `chunk_p` bits
                # A crease is Mountain if v_2(j) is EVEN
                if v2_val % 2 == 0:
                    score_for_this_r_chunk += 1
            else: # v_2 of val is >= chunk_p, meaning val is a multiple of (1 << chunk_p)
                  # This part's v_2 depends on higher bits (i_high)
                next_offsets.append(val >> chunk_p)
        
        # Sort and convert to tuple for memoization key for the recursive call
        next_offsets_tuple = tuple(sorted(set(next_offsets)))
        
        # Recursive call for the higher bits (i_high)
        (sub_even_score, sub_odd_score) = solve(next_offsets_tuple, current_power - chunk_p)

        # Combine scores. We want the full `v_2(current_val)` to be even.
        # `v_2(current_val) = chunk_p + v_2(i_high + B_l)` (where B_l are elements in next_offsets)
        # If `chunk_p` is even: `v_2(current_val)` is even if `v_2(i_high + B_l)` is even.
        # If `chunk_p` is odd: `v_2(current_val)` is even if `v_2(i_high + B_l)` is odd.
        
        if chunk_p % 2 == 0: # chunk_p is even. We want v_2(i_high + B_l) to be even for overall even.
            score_if_current_q_v2_even = score_for_this_r_chunk + sub_even_score
            score_if_current_q_v2_odd = score_for_this_r_chunk + sub_odd_score
        else: # chunk_p is odd. We want v_2(i_high + B_l) to be odd for overall even.
            score_if_current_q_v2_even = score_for_this_r_chunk + sub_odd_score
            score_if_current_q_v2_odd = score_for_this_r_chunk + sub_even_score

        max_even_q_score = max(max_even_q_score, score_if_current_q_v2_even)
        max_odd_q_score = max(max_odd_q_score, score_if_current_q_v2_odd)

    memo[(t_offsets, current_power)] = (max_even_q_score, max_odd_q_score)
    return (max_even_q_score, max_odd_q_score)

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # The function solve returns (max_score_if_overall_q_v2_is_even, max_score_if_overall_q_v2_is_odd)
    # The final i is effectively 'q' at the highest level (current_power=100).
    # We want max score regardless of what v_2(i) is.
    result_even_q_score, result_odd_q_score = solve(A, 100)

    print(max(result_even_q_score, result_odd_q_score))

if __name__ == '__main__':
    main()