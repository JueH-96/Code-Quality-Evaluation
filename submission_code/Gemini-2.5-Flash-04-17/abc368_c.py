import sys

# Read N
N = int(sys.stdin.readline())

# Read the healths H_i
H = list(map(int, sys.stdin.readline().split()))

# Initialize the global timer
total_T = 0

# Process each enemy
for h in H:
    H_current = h
    
    # Calculate the number of full 5-damage cycles (3 attacks each)
    # Each block of 3 attacks deals 5 damage.
    # The number of full 5-damage blocks that can be 'removed' while leaving
    # a remaining health between 1 and 5 is (H_current - 1) // 5.
    num_cycles = (H_current - 1) // 5
    
    # Time taken for these full cycles (each cycle takes 3 turns)
    time_cycles = num_cycles * 3
    
    # Health remaining after these full cycles
    # This value will be between 1 and 5.
    # Equivalent to (H_current - 1) % 5 + 1
    H_rem = H_current - num_cycles * 5
    
    # The current time before starting the remaining damage phase on H_rem.
    # The first attack in this phase happens at total_T (current total) + time_cycles + 1.
    # The damage pattern starting from that time depends on (time + 1) % 3.
    # (total_T + time_cycles + 1) % 3 = (total_T + 3 * num_cycles + 1) % 3 = (total_T + 1) % 3
    # Let rem be the value of (current time + 1) % 3 just before attacking H_rem.
    # The current time is total_T + time_cycles. The next attack is at total_T + time_cycles + 1.
    # So, rem = (total_T + time_cycles + 1) % 3.
    # Since time_cycles is a multiple of 3, time_cycles % 3 == 0.
    # Thus, rem = (total_T + 0 + 1) % 3 = (total_T + 1) % 3.
    rem = (total_T + 1) % 3
    
    # Calculate the number of additional attacks (k_rem) needed for H_rem health
    k_rem = 0
    
    # Based on H_rem (1 to 5) and the starting damage pattern rem (1, 2, or 0 for %3)
    # The damage sequence starting from (total_T + time_cycles + 1) % 3 is:
    # if rem == 1: Pattern (1, 1, 3, 1, 1, 3, ...)
    # if rem == 2: Pattern (1, 3, 1, 1, 3, 1, ...)
    # if rem == 0: Pattern (3, 1, 1, 3, 1, 1, ...)
    
    if rem == 1: # Pattern starts with 1 damage (at times t where t % 3 == 1)
        # Damage sequence: 1, 1, 3, ...
        if H_rem == 1: k_rem = 1 # Needs >= 1 damage. 1st attack gives 1.
        elif H_rem == 2: k_rem = 2 # Needs >= 2 damage. 1st+2nd attacks give 1+1=2.
        else: k_rem = 3           # Needs >= [3,4,5] damage. 1st+2nd+3rd attacks give 1+1+3=5.
    elif rem == 2: # Pattern starts with 1 damage (at times t where t % 3 == 2)
        # Damage sequence: 1, 3, 1, ...
        if H_rem == 1: k_rem = 1 # Needs >= 1 damage. 1st attack gives 1.
        elif H_rem <= 4: k_rem = 2 # Needs >= [2,3,4] damage. 1st+2nd attacks give 1+3=4.
        else: k_rem = 3           # Needs >= 5 damage. 1st+2nd+3rd attacks give 1+3+1=5. (H_rem must be 5 here)
    else: # rem == 0 # Pattern starts with 3 damage (at times t where t % 3 == 0)
        # Damage sequence: 3, 1, 1, ...
        if H_rem <= 3: k_rem = 1 # Needs >= [1,2,3] damage. 1st attack gives 3.
        elif H_rem == 4: k_rem = 2 # Needs >= 4 damage. 1st+2nd attacks give 3+1=4.
        else: k_rem = 3           # Needs >= 5 damage. 1st+2nd+3rd attacks give 3+1+1=5. (H_rem must be 5 here)

    # Total attacks required for the current enemy
    k = time_cycles + k_rem
    
    # Update the global timer by adding the number of turns/attacks spent on this enemy
    total_T += k

# Print the final total time
print(total_T)