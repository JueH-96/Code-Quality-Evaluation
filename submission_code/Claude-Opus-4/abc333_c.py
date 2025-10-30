# YOUR CODE HERE
N = int(input())

# Generate repunits
repunits = []
repunit = 1
# Generate enough repunits - we need to ensure we can get at least 333 different sums
# Since we're looking for sums of 3 repunits, and the largest N is 333,
# we should generate enough repunits to cover this
for i in range(15):  # This should be more than enough
    repunits.append(repunit)
    repunit = repunit * 10 + 1

# Generate all possible sums of exactly three repunits
sums_set = set()
for i in range(len(repunits)):
    for j in range(len(repunits)):
        for k in range(len(repunits)):
            sums_set.add(repunits[i] + repunits[j] + repunits[k])

# Convert to sorted list
sums_list = sorted(list(sums_set))

# Output the N-th smallest (1-indexed)
print(sums_list[N-1])