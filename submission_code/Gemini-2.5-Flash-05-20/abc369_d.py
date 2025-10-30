import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # dp_even: maximum experience points ending with an even number of defeated monsters
    # dp_odd: maximum experience points ending with an odd number of defeated monsters
    
    # Initialize:
    # 0 monsters defeated is an even count, with 0 points.
    dp_even = 0 
    # Cannot have defeated an odd number of monsters before processing any.
    # Use negative infinity to represent an unreachable state.
    dp_odd = -float('inf') 

    for strength in A:
        # Store current dp values (from previous step) before updating for the next step.
        # This is crucial because updates for dp_even and dp_odd for the current monster
        # must be based on the state *before* considering the current monster.
        prev_dp_even = dp_even
        prev_dp_odd = dp_odd

        # Calculate potential new dp_even:
        # 1. Option: Let the current monster go. Experience remains prev_dp_even.
        # 2. Option: Defeat the current monster. To get an even count, we must have had an odd count previously.
        #    Points gained: strength (base) + strength (bonus for even-numbered defeated monster) = 2 * strength.
        #    So, the score would be prev_dp_odd + 2 * strength.
        #    This is only possible if prev_dp_odd was a reachable state (not -inf).
        if prev_dp_odd != -float('inf'):
            dp_even = max(prev_dp_even, prev_dp_odd + 2 * strength)
        else:
            # If prev_dp_odd was -inf, we can't transition to an even state by defeating this monster.
            # So, dp_even can only be from letting the monster go, which is prev_dp_even.
            dp_even = prev_dp_even

        # Calculate potential new dp_odd:
        # 1. Option: Let the current monster go. Experience remains prev_dp_odd.
        # 2. Option: Defeat the current monster. To get an odd count, we must have had an even count previously.
        #    Points gained: strength (base only, no bonus for odd-numbered defeated monster).
        #    So, the score would be prev_dp_even + strength.
        #    prev_dp_even is always a valid state (at worst, it's 0).
        dp_odd = max(prev_dp_odd, prev_dp_even + strength)
    
    # The final answer is the maximum of the two possible states (ending with an even or odd number of defeated monsters).
    print(max(dp_even, dp_odd))

solve()