import itertools

n = int(input())

# Generate repunits up to 20 digits
repunits = []
for k in range(1, 21):
    repunits.append((10**k - 1) // 9)

# Compute all possible sums of three repunits
sums = set()
for combo in itertools.combinations_with_replacement(repunits, 3):
    sums.add(sum(combo))

# Sort the sums and output the N-th smallest
sorted_sums = sorted(sums)
print(sorted_sums[n-1])