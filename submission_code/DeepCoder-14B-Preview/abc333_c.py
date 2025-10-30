import itertools

n = int(input())

# Generate all repunits up to R_100 (100 digits)
repunits = []
k = 1
while k <= 100:
    r = (10**k - 1) // 9
    repunits.append(r)
    k += 1

# Generate all possible triplet sums
sums = set()
for triplet in itertools.product(repunits, repeat=3):
    s = sum(triplet)
    sums.add(s)

# Convert the set to a sorted list
sum_list = sorted(sums)

# Output the N-th smallest sum
print(sum_list[n-1])