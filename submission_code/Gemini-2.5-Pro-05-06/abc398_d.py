import sys

def solve():
    # Read N, R_target, C_target
    N, R_target, C_target = map(int, sys.stdin.readline().split())
    # Read wind string S
    S_winds = sys.stdin.readline().strip()

    # Define wind displacements (delta_row, delta_col)
    wind_displacements = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, 1),
        'W': (0, -1)
    }

    # current_pref_r, current_pref_c will store the coordinates of Pref_t
    # Pref_t = P_1 + ... + P_t (sum of wind vectors from time 1 to t)
    # Before the loop, (current_pref_r, current_pref_c) represents Pref_0 = (0,0).
    current_pref_r = 0
    current_pref_c = 0 

    # seen_prefs stores the coordinates (r,c) of prefix sums encountered so far:
    # Pref_0, Pref_1, ..., Pref_{t-1} (using t as current time step, 1-indexed).
    # We initialize it with Pref_0 = (0,0).
    seen_prefs = set()
    seen_prefs.add((0,0)) # Add Pref_0

    # ans_chars will store the characters '0' or '1' for the output string
    ans_chars = []

    # Iterate for each time step t from 1 to N.
    # S_winds[t_idx] corresponds to the wind P_t where t = t_idx + 1.
    for t_idx in range(N):
        wind_char = S_winds[t_idx]
        dr_wind, dc_wind = wind_displacements[wind_char]

        # Update current_pref_r, current_pref_c from Pref_{t-1} to Pref_t.
        # At loop step t_idx:
        #   current_pref_r,c holds Pref_{t_idx} (which is Pref_{t-1} if t = t_idx+1).
        #   After this update, it holds Pref_{t_idx+1} (which is Pref_t if t = t_idx+1).
        current_pref_r += dr_wind
        current_pref_c += dc_wind
        
        # We are checking if (R_target, C_target) = Pref_t - Pref_{j-1}
        # for some j-1 in {0, 1, ..., t-1}.
        # The set `seen_prefs` contains {Pref_0, ..., Pref_{t-1}} at this point
        # (or {Pref_0, ..., Pref_{t_idx}} in terms of loop variable).
        # So we search for Pref_t - (R_target, C_target) in `seen_prefs`.
        
        # Calculate the coordinates of the Pref_{j-1} we are searching for.
        check_r = current_pref_r - R_target
        check_c = current_pref_c - C_target

        if (check_r, check_c) in seen_prefs:
            ans_chars.append('1')
        else:
            ans_chars.append('0')
            
        # Add Pref_t (which is (current_pref_r, current_pref_c) now) to seen_prefs.
        # This makes it available for future time steps.
        # For the next iteration t_idx+1 (time t+1), seen_prefs will correctly contain Pref_0, ..., Pref_t.
        seen_prefs.add((current_pref_r, current_pref_c))
        
    # Print the result string
    sys.stdout.write("".join(ans_chars) + "
")

if __name__ == '__main__':
    solve()