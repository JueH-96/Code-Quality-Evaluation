# YOUR CODE HERE
n, m = map(int, input().split())
adjacent_pairs = set()

for _ in range(m):
    photo = list(map(int, input().split()))
    for i in range(n - 1):
        person1 = photo[i]
        person2 = photo[i + 1]
        # Always store pairs with smaller person first to avoid duplicates
        if person1 > person2:
            person1, person2 = person2, person1
        adjacent_pairs.add((person1, person2))

# Total possible pairs
total_pairs = n * (n - 1) // 2

# Pairs that may be in a bad mood (never stood next to each other)
bad_mood_pairs = total_pairs - len(adjacent_pairs)

print(bad_mood_pairs)