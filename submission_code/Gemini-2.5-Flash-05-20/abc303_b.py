import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # A set to store all unique pairs of people who stood next to each other
    # in at least one photo.
    # To ensure uniqueness regardless of order (e.g., (1,2) is same as (2,1)),
    # we store pairs in a canonical form: (min_person_id, max_person_id).
    adjacent_pairs_found = set()

    # Process each photo
    for _ in range(M):
        # Read the line of people for the current photo
        photo_line = list(map(int, sys.stdin.readline().split()))
        
        # Iterate through the people in the line to identify all adjacent pairs
        # We iterate up to N-2 because photo_line[N-1] doesn't have a person to its right
        for i in range(N - 1):
            person1 = photo_line[i]
            person2 = photo_line[i+1]
            
            # Create a canonical tuple for the pair (smaller_id, larger_id)
            # This handles cases like (1,2) and (2,1) as the same pair.
            canonical_pair = tuple(sorted((person1, person2)))
            
            # Add this canonical pair to our set. Sets automatically handle uniqueness.
            adjacent_pairs_found.add(canonical_pair)

    # Calculate the total number of distinct pairs of people possible from N people.
    # This is given by the combination formula N choose 2: N * (N - 1) / 2.
    total_possible_pairs = N * (N - 1) // 2

    # The problem states that two people may be in a bad mood if they
    # "did not stand next to each other in any of the photos".
    # This means we need to count pairs that are NOT in our 'adjacent_pairs_found' set.
    # So, we subtract the count of 'non-bad-mood' pairs (those found adjacent)
    # from the total number of possible pairs.
    bad_mood_pairs_count = total_possible_pairs - len(adjacent_pairs_found)

    # Print the final result
    print(bad_mood_pairs_count)

# Call the solve function to execute the program logic
solve()