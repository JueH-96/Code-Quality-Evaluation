# YOUR CODE HERE
n, m = map(int, input().split())

# Set to store all pairs that stood next to each other
adjacent_pairs = set()

for _ in range(m):
    photo = list(map(int, input().split()))
    
    # Find all adjacent pairs in this photo
    for i in range(n - 1):
        person1 = photo[i]
        person2 = photo[i + 1]
        
        # Store pair in canonical order (smaller first)
        pair = (min(person1, person2), max(person1, person2))
        adjacent_pairs.add(pair)

# Total possible pairs
total_pairs = n * (n - 1) // 2

# Pairs that may be in bad mood = total pairs - pairs that stood together
bad_mood_pairs = total_pairs - len(adjacent_pairs)

print(bad_mood_pairs)