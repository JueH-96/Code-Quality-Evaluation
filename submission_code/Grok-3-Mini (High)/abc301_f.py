# YOUR CODE HERE
import sys
sys.setrecursionlimit(1000000)

MOD = 998244353
S = input().strip()
n = len(S)

# States: 0: state 0 (no match), 1: state 2 (matched pair), 2: state 3 (matched pair and lower), 3-28: state 1 with c_idx 0-25
dp_curr = [0] * 29
dp_curr[0] = 1  # Initial state: state 0

for i in range(n):  # For each position in the string
    dp_next = [0] * 29
    for state_id in range(29):
        if dp_curr[state_id] == 0:
            continue
        # Decode current state
        if state_id == 0:
            current_type = 0
        elif state_id == 1:
            current_type = 2
        elif state_id == 2:
            current_type = 3
        elif 3 <= state_id <= 28:
            current_type = 1
            current_c_idx = state_id - 3
        else:
            continue  # Should not happen
        
        char_i = S[i]
        if char_i.isupper():  # Fixed uppercase
            action_type = 'upper'
            action_c_idx = ord(char_i) - ord('A')  # 0-25
            if current_type == 3 and action_type == 'upper':
                # Invalid transition, do nothing
                continue
            # Compute new state
            if current_type == 0:
                new_type = 1
                new_c_idx = action_c_idx
            elif current_type == 1:
                if action_c_idx == current_c_idx:
                    new_type = 2
                else:
                    new_type = 1
                    new_c_idx = action_c_idx
            elif current_type == 2:
                new_type = 2
            # Get new state ID
            if new_type == 1:
                state_id_new = 3 + new_c_idx
            elif new_type == 2:
                state_id_new = 1  # Index for state 2
            dp_next[state_id_new] = (dp_next[state_id_new] + dp_curr[state_id]) % MOD
        elif char_i.islower():  # Fixed lowercase
            action_type = 'lower'
            # Compute new state for 'lower'
            if current_type == 0:
                new_type = 0
            elif current_type == 1:
                new_type = 1
                new_c_idx = current_c_idx
            elif current_type == 2:
                new_type = 3
            elif current_type == 3:
                new_type = 3
            # Get new state ID
            if new_type == 0:
                state_id_new = 0
            elif new_type == 1:
                state_id_new = 3 + new_c_idx
            elif new_type == 3:
                state_id_new = 2  # Index for state 3
            dp_next[state_id_new] = (dp_next[state_id_new] + dp_curr[state_id]) % MOD
        elif char_i == '?':  # Wildcard, can be any letter
            # Add lower action, 26 ways
            if current_type == 0:
                new_type_lower = 0
                state_id_new_lower = 0
            elif current_type == 1:
                new_type_lower = 1
                new_c_idx_lower = current_c_idx
                state_id_new_lower = 3 + current_c_idx
            elif current_type == 2:
                new_type_lower = 3
                state_id_new_lower = 2  # State 3 index
            elif current_type == 3:
                new_type_lower = 3
                state_id_new_lower = 2  # State 3 index
            prod_lower = (dp_curr[state_id] * 26) % MOD
            dp_next[state_id_new_lower] = (dp_next[state_id_new_lower] + prod_lower) % MOD
            
            # Add upper action, if not in state 3
            if current_type != 3:
                if current_type == 0:
                    # All upper d go to state 1 with c=d
                    for new_c_idx in range(26):
                        state_id_new = 3 + new_c_idx
                        dp_next[state_id_new] = (dp_next[state_id_new] + dp_curr[state_id]) % MOD
                elif current_type == 1:
                    # d == current_c_idx: go to state 2 (index 1)
                    dp_next[1] = (dp_next[1] + dp_curr[state_id]) % MOD  # for d == c
                    # d != current_c_idx: go to state 1 with c=d_idx
                    for new_c_idx in range(26):
                        if new_c_idx != current_c_idx:
                            state_id_new = 3 + new_c_idx
                            dp_next[state_id_new] = (dp_next[state_id_new] + dp_curr[state_id]) % MOD
                elif current_type == 2:
                    # All upper d go to state 2 (index 1)
                    prod_upper = (dp_curr[state_id] * 26) % MOD
                    dp_next[1] = (dp_next[1] + prod_upper) % MOD
    # After processing all states for this position, update dp_curr
    dp_curr = dp_next

# After processing all characters, sum over all states
ans = 0
for val in dp_curr:
    ans = (ans + val) % MOD
print(ans)