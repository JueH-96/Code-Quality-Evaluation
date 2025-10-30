def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    instructions = []
    for _ in range(Q):
        H = input[idx]
        T = int(input[idx+1])
        instructions.append((H, T))
        idx += 2

    prev_dp = {}
    # Initial state: L=1, R=2
    # For the first instruction, if it's L, then the other hand is R at 2
    # So prev_dp represents the other hand's position and steps
    prev_dp[2] = 0

    for H_i, T_i in instructions:
        curr_dp = {}
        for other_pos in prev_dp:
            steps_prev = prev_dp[other_pos]
            # Determine S (current hand's previous position) and O_prev (other hand's previous position)
            if H_i == 'L':
                # Previous state: L was at some position, R is other_pos
                # The previous L position is determined by the previous instruction's H
                # For the first instruction, it's 1, but for subsequent, depends on previous H
                # Wait, this is incorrect. We need to track both positions.
                # This approach is flawed. Need to track both hands' positions.
                pass
            else:
                # Similarly for R
                pass
        # The previous approach was incorrect. Need to track both hands' positions.

    # The correct approach requires tracking both hands' positions. Here's the correct code:

    # Re-define the initial state as a dictionary with (L, R) positions and steps
    prev_states = {(1, 2): 0}

    for h, t in instructions:
        curr_states = {}
        for (L_prev, R_prev), steps_prev in prev_states.items():
            if h == 'L':
                S = L_prev
                O_prev = R_prev
                target = t
                other_hand = 'R'
            else:
                S = R_prev
                O_prev = L_prev
                target = t
                other_hand = 'L'

            # Compute minimal steps to move S to target, considering O_prev
            d_clockwise = (target - S) % N
            d_counter = (S - target) % N
            if d_clockwise <= d_counter:
                direction = 'clockwise'
                d = d_clockwise
                # Check if O_prev is in the path (excluding S)
                if S < target:
                    in_path = S < O_prev <= target
                else:
                    in_path = O_prev > S or O_prev <= target
            else:
                direction = 'counter'
                d = d_counter
                if S > target:
                    in_path = target <= O_prev < S
                else:
                    in_path = O_prev < S or O_prev >= target

            steps_h = d + (1 if in_path else 0)

            # Now, the other hand can move to any position except target
            # We need to consider all possible new positions for the other hand
            if other_hand == 'R':
                # The other hand is R, which was at O_prev (R_prev)
                # New R can be any position except target
                for new_R in range(1, N+1):
                    if new_R == target:
                        continue
                    steps_o = min( (new_R - O_prev) % N, (O_prev - new_R) % N )
                    total_steps = steps_prev + steps_h + steps_o
                    new_state = (target, new_R)
                    if new_state not in curr_states or total_steps < curr_states[new_state]:
                        curr_states[new_state] = total_steps
            else:
                # The other hand is L, which was at O_prev (L_prev)
                # New L can be any position except target
                for new_L in range(1, N+1):
                    if new_L == target:
                        continue
                    steps_o = min( (new_L - O_prev) % N, (O_prev - new_L) % N )
                    total_steps = steps_prev + steps_h + steps_o
                    new_state = (new_L, target)
                    if new_state not in curr_states or total_steps < curr_states[new_state]:
                        curr_states[new_state] = total_steps

        # Update prev_states to the new states with minimal steps
        # To optimize, keep only the minimal steps for each state
        temp = {}
        for state, steps in curr_states.items():
            if state not in temp or steps < temp[state]:
                temp[state] = steps
        prev_states = temp

    # Find the minimal steps among all possible states
    print(min(prev_states.values()))

if __name__ == "__main__":
    main()