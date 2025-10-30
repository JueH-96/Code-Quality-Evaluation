# Read the input string S
S = input()

# Extract the numeric part from S (characters from index 3 to the end)
# and convert it to an integer.
# For example, if S is "ABC123", S[3:] is "123", and int("123") is 123.
# If S is "ABC001", S[3:] is "001", and int("001") is 1.
numeric_part_str = S[3:]
N = int(numeric_part_str)

# Check if N falls into the valid ranges.
# Valid numbers are 1-314 (inclusive), 315, and 317-349 (inclusive).
# This is equivalent to N being between 1 and 349 (inclusive),
# but not equal to 316.
if 1 <= N <= 349 and N != 316:
    print("Yes")
else:
    print("No")