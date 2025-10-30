import sys

# Read N
N = int(sys.stdin.readline())

# Read A_i values
# Use map and split for efficient reading of the second line
A = list(map(int, sys.stdin.readline().split()))

# DP state: dp_even, dp_odd
# dp_even: max experience after considering monsters up to current, with an even total number defeated
# dp_odd: max experience after considering monsters up to current, with an odd total number defeated

# Initialize base case (before considering any monsters)
# 0 monsters defeated (even count), 0 experience
dp_even = 0
# Odd number defeated is impossible initially
dp_odd = float('-inf') # Use negative infinity to represent an unreachable state

# Iterate through each monster's strength
for strength in A:
    # Store current DP values before calculating next ones
    prev_dp_even = dp_even
    prev_dp_odd = dp_odd

    # Calculate next state values:
    # To reach an even total defeated count after current monster (strength):
    # This can happen in two ways from the previous state (before considering current monster):
    # 1. We didn't defeat the current monster. The defeated count parity remains even.
    #    The max experience is the max experience from the previous state with an even count: prev_dp_even.
    # 2. We defeated the current monster. The previous defeated count must have been odd to become even now (odd + 1 = even).
    #    The max experience is the max experience from the previous state with an odd count (prev_dp_odd) plus the experience from defeating the current monster.
    #    The current monster is the (odd+1)-th defeated monster. Since (odd+1) is an even number, the experience is 2 * strength.
    #    Total experience: prev_dp_odd + 2 * strength.
    # We take the maximum of these two possibilities.
    # Note: The max function handles the case where prev_dp_odd was float('-inf').
    dp_even = max(prev_dp_even, prev_dp_odd + 2 * strength)

    # To reach an odd total defeated count after current monster (strength):
    # This can happen in two ways from the previous state (before considering current monster):
    # 1. We didn't defeat the current monster. The defeated count parity remains odd.
    #    The max experience is the max experience from the previous state with an odd count: prev_dp_odd.
    # 2. We defeated the current monster. The previous defeated count must have been even to become odd now (even + 1 = odd).
    #    The max experience is the max experience from the previous state with an even count (prev_dp_even) plus the experience from defeating the current monster.
    #    The current monster is the (even+1)-th defeated monster. Since (even+1) is an odd number, the experience is strength.
    #    Total experience: prev_dp_even + strength.
    # We take the maximum of these two possibilities.
    # Note: The max function handles the case where prev_dp_even was float('-inf'). For N >= 1, prev_dp_even starts at 0 and will likely become positive, avoiding -inf unless all A_i=0, which is not possible per constraint 1 <= A_i.
    dp_odd = max(prev_dp_odd, prev_dp_even + strength)


# After iterating through all monsters, the maximum experience is the maximum of ending with
# an even total defeated count (dp_even) or an odd total defeated count (dp_odd).
# Since float('-inf') is smaller than any valid experience (min 0), max will pick the correct one.
print(max(dp_even, dp_odd))