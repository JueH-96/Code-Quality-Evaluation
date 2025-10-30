import sys

# Read N and M from the first line of input
N, M = map(int, sys.stdin.readline().split())

# Store all photo arrangements
# photos_data will be a list of lists, where each inner list represents a photo.
photos_data = []
for _ in range(M):
    # Read each photo's arrangement of people
    photos_data.append(list(map(int, sys.stdin.readline().split())))

# Create an adjacency matrix `are_neighbors` to keep track of pairs of people
# who have stood next to each other in at least one photo.
# `are_neighbors[p1][p2]` will be True if person p1 and person p2
# have ever been adjacent, and False otherwise.
# People are numbered 1 to N. To use 1-based indexing conveniently,
# the matrix size is (N+1)x(N+1).
# Initialize all entries to False, indicating no pairs are known to be neighbors yet.
are_neighbors = [[False for _ in range(N + 1)] for _ in range(N + 1)]

# Iterate through each photo
for photo_idx in range(M):
    current_photo_arrangement = photos_data[photo_idx]
    
    # Iterate through all adjacent pairs of people in the current photo's lineup.
    # For a line of N people, there are N-1 such adjacent pairs.
    # For example, if N=4, pairs are (person at pos 0, person at pos 1), 
    # (person at pos 1, person at pos 2), (person at pos 2, person at pos 3).
    # So, j (0-indexed position) goes from 0 to N-2.
    for j in range(N - 1):
        person1 = current_photo_arrangement[j]
        person2 = current_photo_arrangement[j+1]
        
        # Mark that person1 and person2 have stood next to each other.
        # Since the relationship is symmetric (if p1 is next to p2, p2 is next to p1),
        # update both entries in the adjacency matrix.
        are_neighbors[person1][person2] = True
        are_neighbors[person2][person1] = True

# Now, count the number of pairs of people who may be in a bad mood.
# A pair is in a bad mood if they *never* stood next to each other in *any* photo.
# This corresponds to pairs (p1, p2) for which `are_neighbors[p1][p2]` is still False.
bad_mood_pairs_count = 0

# To avoid double-counting pairs (e.g., (1,2) and (2,1)) and self-pairs (e.g., (1,1)),
# iterate such that p1 < p2.
for p1 in range(1, N + 1):  # Person p1 ranges from 1 to N
    for p2 in range(p1 + 1, N + 1):  # Person p2 ranges from p1+1 to N
        
        # If this pair never stood next to each other
        if not are_neighbors[p1][p2]:
            bad_mood_pairs_count += 1
            
# Print the final count to standard output.
# sys.stdout.write expects a string, so convert the count to string.
# Add a newline character as print() would.
sys.stdout.write(str(bad_mood_pairs_count) + "
")