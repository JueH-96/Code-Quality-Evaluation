from itertools import accumulate

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Calculate prefix sums and their modulo M
prefix_sums = [0] + list(accumulate(A))
prefix_mods = [p % M for p in prefix_sums]

# Count the occurrences of each modulo
mod_counts = {}
for mod in prefix_mods:
    mod_counts[mod] = mod_counts.get(mod, 0) + 1

# Calculate the number of pairs
pairs = 0
for count in mod_counts.values():
    pairs += count * (count - 1) // 2

# Print the result
print(pairs)