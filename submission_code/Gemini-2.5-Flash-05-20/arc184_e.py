import sys

# Optimized input reading
input = sys.stdin.readline

def solve():
    N, M = map(int, input().split())
    
    # Store initial sequences as lists first, then convert to bytearray as needed
    A_raw = []
    for _ in range(N):
        A_raw.append(list(map(int, input().split())))

    MOD = 998244353

    # Function to apply the transformation A'k = (sum_{l=1 to k} Al) mod 2
    # Takes a bytearray, returns a new bytearray
    def apply_operation(seq_bytearray):
        new_seq_bytearray = bytearray(M)
        
        current_sum = 0
        for k in range(M):
            current_sum = (current_sum + seq_bytearray[k]) % 2
            new_seq_bytearray[k] = current_sum
        return new_seq_bytearray

    # Map to store initial sequences (as bytearray) to a list of their original indices.
    # This allows efficient lookup of which A_j original sequences match a computed state.
    initial_states_map = {}
    for j in range(N):
        seq_bytearray = bytearray(A_raw[j])
        # Use .get() to append to existing list or create a new one
        initial_states_map.setdefault(seq_bytearray, []).append(j)

    total_f_sum = 0

    for i in range(N):
        # Dictionary to store (sequence_state -> x_value) for A_i's path.
        # This helps detect cycles and find the smallest x for a state.
        visited_states_for_Ai = {}
        
        # Start with the original A_i sequence
        current_A_seq_bytearray = bytearray(A_raw[i]) 
        current_x = 0
        
        # Simulate applying operations until a state repeats
        while True:
            # Check if current state has been visited in this A_i's path
            if current_A_seq_bytearray in visited_states_for_Ai:
                break # Cycle detected, stop for this A_i
            
            # Record the current state and the x-value (number of operations to reach it)
            visited_states_for_Ai[current_A_seq_bytearray] = current_x
            
            # Check if this current state matches any initial A_j sequence
            if current_A_seq_bytearray in initial_states_map:
                # If it matches, add current_x to the total sum for relevant j's
                for j_orig_idx in initial_states_map[current_A_seq_bytearray]:
                    if j_orig_idx >= i: # Problem requires sum for j >= i
                        total_f_sum = (total_f_sum + current_x) % MOD
            
            # Compute the next state by applying the operation
            current_A_seq_bytearray = apply_operation(current_A_seq_bytearray)
            # Increment x-value for the next state
            current_x += 1

    # Print the final sum
    sys.stdout.write(str(total_f_sum) + "
")

solve()