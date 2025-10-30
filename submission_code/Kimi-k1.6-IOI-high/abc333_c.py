n = int(input())

# Generate repunits up to a sufficiently large length (e.g., 40)
repunits = []
for k in range(1, 41):  # Generates R1 to R40
    repunits.append((10**k - 1) // 9)

sums = set()

# Compute all possible sums of three repunits
for i in range(len(repunits)):
    for j in range(len(repunits)):
        for k in range(len(repunits)):
            s = repunits[i] + repunits[j] + repunits[k]
            sums.add(s)

# Sort the unique sums and find the N-th smallest
sorted_sums = sorted(sums)
print(sorted_sums[n-1])