# Read the input string
S = input().strip()

# List of valid strings
valid_strings = ["ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"]

# Check if the input string is in the list of valid strings
if S in valid_strings:
    print("Yes")
else:
    print("No")