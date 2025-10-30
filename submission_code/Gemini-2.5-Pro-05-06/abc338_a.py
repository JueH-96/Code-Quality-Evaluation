# Read the input string
S = input()

# Check if the first character is uppercase
# S is guaranteed to be non-empty, so S[0] is always valid.
first_char_ok = S[0].isupper()

# Check if all other characters (from the second character onwards) are lowercase.
# S[1:] creates a slice of the string from index 1 to the end.
# If S has only one character, S[1:] will be an empty string.
# The all() function returns True if the iterable is empty.
# This correctly handles the case where there are no "other characters".
other_chars_ok = all(char.islower() for char in S[1:])

# If both conditions are met, print "Yes"
if first_char_ok and other_chars_ok:
    print("Yes")
else:
    # Otherwise, print "No"
    print("No")