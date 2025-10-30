# YOUR CODE HERE
import sys

# Read N (number of people) and M (number of photos)
N, M = map(int, sys.stdin.readline().split())

# Initialize a set to store pairs of people who stood next to each other in any photo.
# Using frozenset makes the pair hashable and treats {a, b} the same as {b, a}.
adjacent_pairs = set()

# Process each photo
for _ in range(M):
    # Read the list of people in the current photo
    # The input uses 1-based indexing for people (1 to N)
    photo = list(map(int, sys.stdin.readline().split()))
    
    # Iterate through the list to find adjacent people in this photo.
    # In a list of N people, there are N-1 adjacent pairs: (photo[0], photo[1]), ..., (photo[N-2], photo[N-1]).
    for i in range(N - 1):
        person1 = photo[i]
        person2 = photo[i+1]
        
        # Add the unordered pair of these two adjacent people to our set.
        # frozenset is used because set elements must be hashable
        adjacent_pairs.add(frozenset({person1, person2}))

# Now, count the number of pairs of people who were never adjacent in any photo.
# These are the pairs who "may be in a bad mood".
bad_mood_pairs_count = 0

# Iterate through all possible unique pairs of people (p1, p2) where p1 < p2.
# People are numbered from 1 to N.
for p1 in range(1, N + 1):
    # Start p2 from p1 + 1 to ensure we only consider each pair once and p1 != p2.
    for p2 in range(p1 + 1, N + 1):
        # Form the pair as a frozenset for checking against our set of adjacent pairs.
        pair = frozenset({p1, p2})
        
        # If this pair was NOT found in the set of adjacent pairs from any photo,
        # it means they never stood next to each other.
        if pair not in adjacent_pairs:
            bad_mood_pairs_count += 1

# Print the final count of pairs who may be in a bad mood.
print(bad_mood_pairs_count)