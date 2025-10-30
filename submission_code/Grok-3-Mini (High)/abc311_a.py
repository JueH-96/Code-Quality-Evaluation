# Read input from stdin
N = int(input())
S = input()

# Initialize a set to keep track of seen characters
seen = set()

# Iterate through the string with index
for i, char in enumerate(S):
    seen.add(char)
    # Check if all three characters are seen
    if len(seen) == 3:
        # Print the number of characters checked (1-based index)
        print(i + 1)
        # Exit the loop as the condition is satisfied
        break