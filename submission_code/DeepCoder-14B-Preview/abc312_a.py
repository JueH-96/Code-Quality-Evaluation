# Read the input string
s = input().strip()

# Define the set of valid strings
valid = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if the input string is in the valid set
if s in valid:
    print("Yes")
else:
    print("No")