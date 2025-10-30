# Read the input string from standard input
S = input().strip()

# Define the set of valid strings
valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

# Check if S is in the set of valid strings and print Yes or No accordingly
if S in valid_strings:
    print("Yes")
else:
    print("No")