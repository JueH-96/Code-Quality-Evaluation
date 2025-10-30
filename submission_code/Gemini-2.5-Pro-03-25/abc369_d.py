# YOUR CODE HERE
import sys

def solve():
    # Read the number of monsters, N
    N = int(sys.stdin.readline())
    
    # Read the strengths of the N monsters into a list A
    # Check if N=0 first to avoid reading A if there are no monsters. Although N>=1 constraint means this is not needed.
    if N == 0:
        print(0)
        return
    
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize DP states:
    # dp_even stores the maximum experience points achievable after considering some prefix of monsters,
    # such that the total number of defeated monsters so far is EVEN.
    # Start with 0 points having defeated 0 monsters (an even number).
    dp_even = 0 
    
    # dp_odd stores the maximum experience points achievable after considering some prefix of monsters,
    # such that the total number of defeated monsters so far is ODD.
    # Initialize with -1 to represent an impossible/unreached state. Since all monster strengths A_i >= 1,
    # any achievable score with an odd number of defeats must be at least 1. Thus, -1 safely represents
    # impossibility compared to achievable non-negative scores.
    dp_odd = -1   

    # Iterate through each monster's strength `ai` in the list `A`
    for ai in A: 
        
        # We need to calculate the new DP states based on the values from the *previous* step.
        # Use temporary variables `new_dp_even` and `new_dp_odd` to store the results for the current step
        # before updating the main `dp_even` and `dp_odd` variables for the next iteration.
        
        # --- Calculate the new maximum score ending with an EVEN number of defeats ---
        
        # Option 1: Do not defeat the current monster `ai`. 
        # If we don't defeat `ai`, the parity of defeated monsters doesn't change.
        # The maximum score ending with an even count remains the same as the previous step's `dp_even`.
        current_max_even = dp_even 
        
        # Option 2: Defeat the current monster `ai`.
        # To end up with an EVEN count after defeating `ai`, we must have started this step with an ODD count.
        # Check if the `dp_odd` state was reachable (i.e., `dp_odd != -1`).
        if dp_odd != -1:
            # If it was reachable, defeating `ai` makes it the k-th defeated monster where k = (previous odd count + 1), which is an even number.
            # An even-numbered defeated monster provides `ai` base experience + `ai` bonus experience = `2 * ai` total points for this monster.
            # The total score would be the score from the previous odd state (`dp_odd`) plus `2 * ai`.
            potential_even_from_odd = dp_odd + 2 * ai
            # Update `current_max_even` if this path (defeating `ai`) yields a higher score than not defeating it.
            current_max_even = max(current_max_even, potential_even_from_odd)
        
        # Store the maximum possible score ending in an even state after considering `ai`.
        new_dp_even = current_max_even

        # --- Calculate the new maximum score ending with an ODD number of defeats ---

        # Option 1: Do not defeat the current monster `ai`.
        # If we don't defeat `ai`, the parity of defeated monsters doesn't change.
        # The maximum score ending with an odd count remains the same as the previous step's `dp_odd`.
        current_max_odd = dp_odd

        # Option 2: Defeat the current monster `ai`.
        # To end up with an ODD count after defeating `ai`, we must have started this step with an EVEN count.
        # The `dp_even` state is always reachable (it starts at 0 and can only increase or stay non-negative).
        # Defeating `ai` makes it the k-th defeated monster where k = (previous even count + 1), which is an odd number.
        # An odd-numbered defeated monster provides `ai` base experience points only.
        # The total score would be the score from the previous even state (`dp_even`) plus `ai`.
        potential_odd_from_even = dp_even + ai
        # Update `current_max_odd` if this path (defeating `ai`) yields a higher score than not defeating it.
        current_max_odd = max(current_max_odd, potential_odd_from_even)

        # Store the maximum possible score ending in an odd state after considering `ai`.
        new_dp_odd = current_max_odd

        # Update the DP states (`dp_even`, `dp_odd`) for the next iteration using the computed values for this step.
        dp_even = new_dp_even
        dp_odd = new_dp_odd

    # After iterating through all N monsters, the overall maximum experience points achievable
    # is the maximum value between the final `dp_even` state and the final `dp_odd` state.
    # The `max` function handles the case where `dp_odd` might still be -1 (meaning no path ending with an odd number of defeats was ever optimal or possible).
    print(max(dp_even, dp_odd))

# Execute the solve function to run the DP algorithm and print the result.
solve()