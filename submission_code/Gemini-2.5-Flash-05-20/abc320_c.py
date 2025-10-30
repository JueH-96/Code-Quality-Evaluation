import math

def solve():
    # Read M (length of reel strings)
    M = int(input())
    # Read the three reel strings S_1, S_2, S_3
    S = [input() for _ in range(3)]

    # Initialize min_overall_time to positive infinity. This will store the minimum
    # time found across all possible target characters.
    min_overall_time = math.inf

    # Iterate through all possible target digits (0 to 9)
    for target_char_code in range(10):
        target_char = str(target_char_code)

        # reel_possible_times[r_idx] will store a sorted list of distinct absolute times
        # where reel 'r_idx' can display 'target_char'.
        reel_possible_times = [[] for _ in range(3)]

        # Populate reel_possible_times for each reel
        for r_idx in range(3): # For each of the three reels (0, 1, 2)
            # Iterate through each position (index) on the current reel
            for i in range(M):
                if S[r_idx][i] == target_char:
                    # If target_char is found at index 'i', then reel 'r_idx' can display it at times:
                    # 'i' (0 full cycles), 'i + M' (1 full cycle), 'i + 2 * M' (2 full cycles).
                    # Considering up to 2 full cycles (i.e., times up to M-1 + 2*M = 3*M-1) is sufficient
                    # to find the minimal max time for distinct reel stops.
                    reel_possible_times[r_idx].append(i)
                    reel_possible_times[r_idx].append(i + M)
                    reel_possible_times[r_idx].append(i + 2 * M)
            
            # Remove duplicate times (e.g., if a char appears at multiple positions that align, or M=1)
            # and sort the list of times. Also, filter out times that are outside our maximum search range (3*M).
            # Note: `t < 3 * M` ensures we only keep times up to `3*M - 1`.
            reel_possible_times[r_idx] = sorted(list(set(t for t in reel_possible_times[r_idx] if t < 3 * M)))

        # If the target_char does not appear on ALL of the reels (meaning one of the lists is empty),
        # then it's impossible to make all reels display this character. Skip to the next target_char.
        if not reel_possible_times[0] or not reel_possible_times[1] or not reel_possible_times[2]:
            continue

        # Initialize the minimum max time found for the current target_char to infinity.
        current_min_max_T_for_char = math.inf

        # Iterate through all combinations of possible stop times (t0, t1, t2)
        # for reels 0, 1, and 2 respectively, to find the minimum max(t0, t1, t2)
        # such that t0, t1, t2 are distinct.

        for t0 in reel_possible_times[0]:
            # Pruning 1: If current t0 is already greater than or equal to the best global time found so far,
            # any combination involving this t0 (and potentially larger t1, t2) cannot be better.
            # Since reel_possible_times are sorted, we can break from this loop.
            if t0 >= min_overall_time:
                break 

            for t1 in reel_possible_times[1]:
                # Constraint: All stop times (t0, t1, t2) must be distinct.
                if t0 == t1:
                    continue
                
                # Pruning 2: If the maximum of t0 and t1 is already greater than or equal to the
                # best global time found so far, then no choice of t2 can improve the result.
                # Break from this t1 loop.
                if max(t0, t1) >= min_overall_time:
                    break 

                for t2 in reel_possible_times[2]:
                    # Constraint: All stop times (t0, t1, t2) must be distinct.
                    if t0 == t2 or t1 == t2:
                        continue
                    
                    # Calculate the maximum time for the current combination (t0, t1, t2).
                    candidate_max_T = max(t0, t1, t2)
                    
                    # Update the minimum max time for the current target_char.
                    current_min_max_T_for_char = min(current_min_max_T_for_char, candidate_max_T)

                    # Pruning 3: If candidate_max_T (which uses the current t2) is already
                    # greater than or equal to the best max time found for this target_char so far
                    # (current_min_max_T_for_char), then further increasing t2 will only yield
                    # equal or larger max times. So, break from this t2 loop.
                    if candidate_max_T >= current_min_max_T_for_char: 
                        break
        
        # After checking all relevant combinations for the current target_char,
        # update the overall minimum time found.
        min_overall_time = min(min_overall_time, current_min_max_T_for_char)

    # After checking all target characters ('0' through '9'),
    # if min_overall_time is still infinity, it means no solution was found.
    if min_overall_time == math.inf:
        print(-1)
    else:
        print(min_overall_time)

# Call the solve function to run the program
solve()