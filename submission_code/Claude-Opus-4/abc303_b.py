# YOUR CODE HERE
N, M = map(int, input().split())

# Set to store all pairs that were adjacent in at least one photo
adjacent_pairs = set()

# Process each photo
for _ in range(M):
    photo = list(map(int, input().split()))
    
    # For each adjacent pair in this photo
    for j in range(N - 1):
        person1 = photo[j]
        person2 = photo[j + 1]
        
        # Store the pair in sorted order to avoid duplicates
        pair = tuple(sorted([person1, person2]))
        adjacent_pairs.add(pair)

# Total possible pairs
total_pairs = N * (N - 1) // 2

# Pairs that may be in bad mood = total pairs - pairs that were adjacent
bad_mood_pairs = total_pairs - len(adjacent_pairs)

print(bad_mood_pairs)