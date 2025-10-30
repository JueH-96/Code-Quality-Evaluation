# Read the input string from stdin
S = input().strip()

# Define the list of valid strings
valid_strings = ["ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"]

# Check if the input string is in the list of valid strings
if S in valid_strings:
    # If it is, print "Yes"
    print("Yes")
else:
    # If it's not, print "No"
    print("No")