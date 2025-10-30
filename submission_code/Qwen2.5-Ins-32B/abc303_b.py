import sys

def read_ints(): return map(int, sys.stdin.readline().strip().split())

N, M = read_ints()
photos = [list(read_ints()) for _ in range(M)]

# Initialize a set to keep track of pairs that stood next to each other
next_to_each_other = set()

# Process each photo
for photo in photos:
    for i in range(N-1):
        # Add both (a, b) and (b, a) to the set to avoid duplicates
        next_to_each_other.add((photo[i], photo[i+1]))
        next_to_each_other.add((photo[i+1], photo[i]))

# Calculate the total number of pairs
total_pairs = N * (N - 1) // 2

# Calculate the number of pairs that stood next to each other
next_to_each_other_count = len(next_to_each_other) // 2

# The number of pairs that may be in a bad mood
bad_mood_pairs = total_pairs - next_to_each_other_count

print(bad_mood_pairs)