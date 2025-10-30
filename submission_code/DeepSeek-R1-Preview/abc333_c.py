n = int(input())

# Generate repunits up to 40 digits
repunits = []
for k in range(1, 41):
    repunit = (10**k - 1) // 9
    repunits.append(repunit)

sums = set()

# Generate all possible sums of three repunits with i <= j <= k
for i in range(len(repunits)):
    for j in range(i, len(repunits)):
        for k in range(j, len(repunits)):
            s = repunits[i] + repunits[j] + repunits[k]
            sums.add(s)

# Convert the set to a sorted list
sums_list = sorted(sums)

# Output the N-th smallest sum
print(sums_list[n-1])