import sys

# Read the number of sticks
N = int(sys.stdin.readline())

# Use a set to store the canonical representation of unique sticks.
# The canonical representation of a stick S is the lexicographically
# smaller string between S and its reverse.
unique_sticks = set()

# Process each stick
for _ in range(N):
    S = sys.stdin.readline().strip() # Read the string representing the stick
    R = S[::-1] # Get the reverse of the string
    
    # Determine the canonical form: the lexicographically smaller of S and R
    canonical_S = min(S, R)
    
    # Add the canonical form to the set. Sets automatically handle duplicates.
    unique_sticks.add(canonical_S)

# The number of different sticks is the number of unique canonical forms
result = len(unique_sticks)

# Print the result
print(result)