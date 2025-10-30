import sys

# Read X, Y, Z
line1 = sys.stdin.readline().split()
X = int(line1[0])
Y = int(line1[1])
Z = int(line1[2])

# Read S
S = sys.stdin.readline().strip()

# dp_off: min cost to type up to current character, ending with Caps Lock off
# dp_on: min cost to type up to current character, ending with Caps Lock on

# Initial state: typed 0 characters, Caps Lock is off
# Cost to type 0 chars ending CL off is 0.
# Cost to type 0 chars ending CL on is Z (press Caps Lock once).
dp_off = 0
dp_on = Z

# Iterate through each character in S
for char in S:
    # cost_type_off: Cost to type the target character 'char' when the current CL state is OFF.
    # cost_type_on: Cost to type the target character 'char' when the current CL state is ON.

    if char == 'a':
        # To type 'a' when CL is OFF, use 'a' key (cost X)
        # To type 'a' when CL is ON, use Shift+'a' (cost Y)
        cost_type_off = X
        cost_type_on = Y
    else: # char == 'A'
        # To type 'A' when CL is OFF, use Shift+'a' (cost Y)
        # To type 'A' when CL is ON, use 'a' key (cost X)
        cost_type_off = Y
        cost_type_on = X

    # Calculate the next DP states for typing the current character `char`

    # next_dp_off: min cost to type up to `char`, ending with CL off
    # To end with CL off after typing `char`, we must use the key action that produces `char` when CL is OFF (cost cost_type_off).
    # This action must be performed after potentially changing the CL state.
    # Option 1: Come from previous state where `S[0...i-1]` was typed ending CL off (cost dp_off).
    #           Stay CL off, type `char` (cost cost_type_off). Total: dp_off + cost_type_off.
    # Option 2: Come from previous state where `S[0...i-1]` was typed ending CL on (cost dp_on).
    #           Toggle CL (cost Z) to become CL off, type `char` (cost cost_type_off). Total: dp_on + Z + cost_type_off.
    next_dp_off = min(dp_off + cost_type_off, dp_on + Z + cost_type_off)


    # next_dp_on: min cost to type up to `char`, ending with CL on
    # To end with CL on after typing `char`, we must use the key action that produces `char` when CL is ON (cost cost_type_on).
    # This action must be performed after potentially changing the CL state.
    # Option 1: Come from previous state where `S[0...i-1]` was typed ending CL off (cost dp_off).
    #           Toggle CL (cost Z) to become CL on, type `char` (cost cost_type_on). Total: dp_off + Z + cost_type_on.
    # Option 2: Come from previous state where `S[0...i-1]` was typed ending CL on (cost dp_on).
    #           Stay CL on, type `char` (cost cost_type_on). Total: dp_on + cost_type_on.
    next_dp_on = min(dp_off + Z + cost_type_on, dp_on + cost_type_on)

    # Update DP states for the next iteration (next character)
    dp_off = next_dp_off
    dp_on = next_dp_on

# After processing all characters, the minimum total cost is the minimum
# of the cost ending in CL off state or the cost ending in CL on state.
result = min(dp_off, dp_on)

# Print the result
print(result)