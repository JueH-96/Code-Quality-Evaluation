import sys

# Read the input string from standard input
s = sys.stdin.readline().strip()

# To check if the characters in S can be rearranged to form "ABC",
# we can sort the characters in S and check if the result is "ABC".
# This is because "ABC" is the sorted version of its characters,
# and any string that can be rearranged to "ABC" must contain exactly one 'A', one 'B', and one 'C'.

# Convert the string to a list of characters
char_list = list(s)

# Sort the list of characters
char_list.sort()

# Join the sorted list back into a string
sorted_s = "".join(char_list)

# Check if the sorted string is equal to "ABC"
if sorted_s == "ABC":
    print("Yes")
else:
    print("No")