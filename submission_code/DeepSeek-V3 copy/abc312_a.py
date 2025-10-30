# Read the input string
S = input().strip()

# List of valid strings
valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if S is in the valid strings
if S in valid_strings:
    print("Yes")
else:
    print("No")