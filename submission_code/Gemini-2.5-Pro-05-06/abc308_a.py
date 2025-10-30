# Read the eight integers from standard input
S = list(map(int, input().split()))

# Initialize the result variable. Assume "Yes" until a condition is violated.
result = "Yes"

# Condition 1: The sequence must be monotonically non-decreasing.
# S_1 <= S_2 <= ... <= S_8
# There are 8 elements, so S has indices 0 through 7.
# We need to check pairs (S[0],S[1]), (S[1],S[2]), ..., (S[6],S[7]).
# This means looping i from 0 to 6. range(7) or range(len(S)-1) achieves this.
for i in range(len(S) - 1):
    if S[i] > S[i+1]:
        result = "No"
        break  # Exit this loop as condition is already violated

# Only proceed to check other conditions if Condition 1 was met.
# If result is already "No", these checks will be skipped.
if result == "Yes":
    # Conditions 2 and 3 apply to each individual number S_i.
    # We iterate through each element in S.
    for s_val in S:
        # Condition 2: S_i must be between 100 and 675, inclusive.
        if not (100 <= s_val <= 675):
            result = "No"
            break  # Exit this loop as condition is already violated for this s_val

        # Condition 3: S_i must be a multiple of 25.
        # This means s_val modulo 25 should be 0.
        if s_val % 25 != 0:
            result = "No"
            break  # Exit this loop as condition is already violated for this s_val

# Print the final result
print(result)