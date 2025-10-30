import sys

def solve():
    N = int(sys.stdin.readline())
    
    # Use a set to store the unique canonical representations of the sticks.
    unique_sticks = set()
    
    for _ in range(N):
        # Read the string for the current stick.
        # .strip() is used to remove the trailing newline character.
        s = sys.stdin.readline().strip()
        
        # Calculate the reverse of the string.
        s_rev = s[::-1]
        
        # Determine the canonical form: the lexicographically smaller of s and s_rev.
        # This ensures that "abc" and "cba" both map to "abc".
        canonical_s = min(s, s_rev)
        
        # Add the canonical form to the set.
        # If it's already present, the set's size won't change.
        unique_sticks.add(canonical_s)
        
    # The number of different sticks is the total number of unique canonical forms.
    print(len(unique_sticks))

# Call the solve function to run the program.
solve()