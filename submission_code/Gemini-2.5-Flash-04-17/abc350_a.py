# Read the input string
S = input()

# Extract the last three characters, which are the digits
digits_str = S[3:]

# Convert the digits string to an integer
contest_number = int(digits_str)

# Check if the contest number is in the allowed range and not the excluded number
# The allowed range is 1 to 349, excluding 316.
if 1 <= contest_number <= 349 and contest_number != 316:
    print("Yes")
else:
    print("No")