# Read the input string
S = input().strip()

# List of valid strings to check against
valid = ['ACE', 'BDF', 'CEG', 'DFA', 'EGB', 'FAC', 'GBD']

# Check if S is in the valid list
if S in valid:
    print("Yes")
else:
    print("No")