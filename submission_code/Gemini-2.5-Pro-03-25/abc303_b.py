# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read N (number of people) and M (number of photos) from the first line of input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the M photos. Each photo is represented as a list of N integers (people IDs).
    # Store these photo line-ups in a list of lists.
    photos = []
    for _ in range(M):
        photos.append(list(map(int, sys.stdin.readline().split())))

    # Initialize a set to keep track of unique pairs of people who have stood next to each other
    # in at least one photo. Using a set automatically handles duplicates efficiently.
    stood_together_pairs = set()

    # Iterate through each photo provided
    for i in range(M):
        current_photo = photos[i]
        # Iterate through adjacent positions in the current photo's line-up.
        # The loop goes from the first person (index 0) up to the second to last person (index N-2).
        # This covers all adjacent pairs: (person at j, person at j+1).
        for j in range(N - 1):
            # Get the IDs of the two adjacent people
            p1 = current_photo[j]
            p2 = current_photo[j+1]
            
            # Create a canonical representation for the pair {p1, p2}.
            # We use a tuple (min(p1, p2), max(p1, p2)) to represent the unordered pair {p1, p2}.
            # This ensures that the pair {x, y} is stored consistently as (x, y) if x < y,
            # regardless of the order they appeared adjacent in the photo (e.g., x then y, or y then x).
            if p1 < p2:
                pair = (p1, p2)
            else:
                # If p1 > p2, swap them to maintain the canonical order (smaller_id, larger_id)
                pair = (p2, p1)
            # Since each photo is a permutation of 1..N, p1 will never be equal to p2.
            
            # Add the identified adjacent pair to the set.
            # If this pair has already been added (from this photo or another), the set's add operation
            # effectively does nothing, maintaining uniqueness.
            stood_together_pairs.add(pair)

    # Calculate the total number of distinct pairs possible among N people.
    # This is given by the combination formula "N choose 2", which equals N * (N - 1) / 2.
    # Use integer division // to ensure the result is an integer.
    total_possible_pairs = N * (N - 1) // 2
    
    # The number of pairs that stood together at least once is the total number of unique pairs
    # collected in the set `stood_together_pairs`.
    num_stood_together = len(stood_together_pairs)
    
    # The problem asks for the number of pairs who "may be in a bad mood".
    # This condition applies to pairs who *never* stood next to each other in any of the M photos.
    # This count is obtained by subtracting the number of pairs that did stand together at least once
    # from the total number of possible pairs.
    bad_mood_pairs_count = total_possible_pairs - num_stood_together
    
    # Print the final calculated count to standard output.
    print(bad_mood_pairs_count)

# Call the solve function to execute the main logic of reading input and finding the answer.
solve()