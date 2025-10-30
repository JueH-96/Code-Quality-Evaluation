N = int(input())

# Generate repunits up to 13 digits
repunits = []
for i in range(1, 14):
    repunit = int('1' * i)
    repunits.append(repunit)

# Generate all possible sums of exactly three repunits
sums = set()
for i in range(len(repunits)):
    for j in range(i, len(repunits)):
        for k in range(j, len(repunits)):
            sums.add(repunits[i] + repunits[j] + repunits[k])

# Sort the sums and return the N-th one
sorted_sums = sorted(sums)
print(sorted_sums[N - 1])