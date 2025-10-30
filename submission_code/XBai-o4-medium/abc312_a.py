# Read input from standard input
S = input().strip()

# Define the set of valid strings
valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if S is in the valid set and print the result
print("Yes" if S in valid_strings else "No")