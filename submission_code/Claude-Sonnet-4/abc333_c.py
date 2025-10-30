n = int(input())

# Generate repunits
repunits = []
repunit = 1
# Generate enough repunits - we need to be careful about the upper bound
# Since we're looking for the 333rd number at most, and the largest could be quite big
# Let's generate repunits up to a reasonable length
for i in range(1, 15):  # This should be more than enough
    repunit = (10**i - 1) // 9
    repunits.append(repunit)

# Generate all possible sums of exactly three repunits
sums = set()

# Use three nested loops to get all combinations with repetition
for i in range(len(repunits)):
    for j in range(len(repunits)):
        for k in range(len(repunits)):
            sum_val = repunits[i] + repunits[j] + repunits[k]
            sums.add(sum_val)

# Convert to sorted list
sorted_sums = sorted(list(sums))

# Output the N-th smallest (1-indexed)
print(sorted_sums[n-1])