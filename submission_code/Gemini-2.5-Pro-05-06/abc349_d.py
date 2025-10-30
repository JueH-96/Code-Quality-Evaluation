import sys

def solve():
    L, R = map(int, sys.stdin.readline().split())

    ans_pairs = []
    current_L = L

    while current_L < R:
        best_i = -1
        
        # Iterate i from high to low to find the largest 2^i step.
        # Max i can be 60 because R <= 2^60.
        # If L=0, R=2^60, then i=60 gives segment (0, 2^60).
        for i in range(60, -1, -1): # i from 60 down to 0
            pow2_i = 1 << i
            
            # Condition 1: current_L must be a multiple of pow2_i.
            # current_L % pow2_i == 0 handles current_L = 0 correctly, as 0 % X == 0 for X > 0.
            if current_L % pow2_i == 0:
                # Condition 2: The segment must not extend beyond R.
                if current_L + pow2_i <= R:
                    best_i = i
                    break # Found the largest i satisfying both conditions
        
        # This assertion should hold because for i=0, pow2_i=1.
        # current_L % 1 == 0 is always true.
        # As long as current_L < R, current_L + 1 <= R is true (unless current_L = R, but loop ends).
        # So i=0 is always a possibility if no larger i works.
        assert best_i != -1, "Internal error: Failed to find a valid i."

        chosen_segment_length = 1 << best_i
        ans_pairs.append((current_L, current_L + chosen_segment_length))
        current_L += chosen_segment_length

    sys.stdout.write(str(len(ans_pairs)) + "
")
    for l_val, r_val in ans_pairs:
        sys.stdout.write(str(l_val) + " " + str(r_val) + "
")

solve()