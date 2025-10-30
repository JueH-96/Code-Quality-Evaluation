# Read the input string
S = input().strip()

# Define the set of valid strings
valid = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if S is in the valid set and print the result
print("Yes" if S in valid else "No")