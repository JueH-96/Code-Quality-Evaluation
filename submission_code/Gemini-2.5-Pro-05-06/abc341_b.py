import sys

# Read N
N = int(sys.stdin.readline())
# Read initial amounts of currencies A_1, ..., A_N
# Store them in a 0-indexed list `A_amounts` where A_amounts[i] is currency of country i+1.
A_amounts = list(map(int, sys.stdin.readline().split()))

# There are N-1 types of conversions.
# The loop variable `i` (0 to N-2) corresponds to converting
# currency of country `i+1` (stored in A_amounts[i])
# to currency of country `i+2` (stored in A_amounts[i+1]).
for i in range(N - 1):
    # Read S_val and T_val for the current conversion type.
    # These are S_{i+1} and T_{i+1} in 1-indexed problem statement.
    s_val, t_val = map(int, sys.stdin.readline().split())
    
    # Calculate how many times Takahashi can perform this conversion.
    # If A_amounts[i] < s_val, then num_exchanges will be 0.
    num_exchanges = A_amounts[i] // s_val
    
    # Add the T_val * num_exchanges units of currency to country i+2.
    A_amounts[i+1] += num_exchanges * t_val
    
    # The amount A_amounts[i] is depleted, but its remaining value is not needed
    # for maximizing A_amounts[N-1] as it cannot be converted further "upwards".
    # A_amounts[i] -= num_exchanges * s_val # This line is not necessary.

# The result is the amount of currency for country N, which is A_amounts[N-1].
print(A_amounts[N-1])