# Read input from standard input (usually you'd use input() in Python)
S = input().strip()

# Define the set of valid strings
valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if the input string is in the set of valid strings
if S in valid_strings:
    print("Yes")
else:
    print("No")