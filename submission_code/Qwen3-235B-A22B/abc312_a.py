# Read the input string
s = input().strip()

# Define the set of valid strings
valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if the input is in the valid set and print the result
print("Yes" if s in valid_strings else "No")