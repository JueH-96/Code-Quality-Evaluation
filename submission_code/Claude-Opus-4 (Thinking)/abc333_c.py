# YOUR CODE HERE
n = int(input())

# Generate repunits up to 13 digits
repunits = []
for k in range(1, 14):
    repunit = (10**k - 1) // 9
    repunits.append(repunit)

# Generate all possible sums of exactly three repunits
sums = []
for i in range(len(repunits)):
    for j in range(i, len(repunits)):
        for k in range(j, len(repunits)):
            sum_val = repunits[i] + repunits[j] + repunits[k]
            sums.append(sum_val)

# Sort the sums and print the n-th smallest
sums.sort()
print(sums[n-1])