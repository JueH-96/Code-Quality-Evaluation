def count_bad_mood_pairs(N, M, photos):
    # Create a set to keep track of adjacent pairs
    adjacent_pairs = set()

    # Iterate through each photo
    for photo in photos:
        for j in range(N - 1):
            # Create pairs of adjacent people
            pair = (min(photo[j], photo[j + 1]), max(photo[j], photo[j + 1]))
            adjacent_pairs.add(pair)

    # Total pairs of people
    total_pairs = N * (N - 1) // 2

    # Pairs that are not adjacent
    non_adjacent_pairs = total_pairs - len(adjacent_pairs)

    return non_adjacent_pairs

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
photos = [list(map(int, line.split())) for line in data[1:M + 1]]

# Get the result
result = count_bad_mood_pairs(N, M, photos)

# Print the result
print(result)