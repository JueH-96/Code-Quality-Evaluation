n = int(input())
max_length = 15  # Sufficient to cover up to the required N=333 case

repunits = []
for k in range(1, max_length + 1):
    rep = (10**k - 1) // 9
    repunits.append(rep)

sums = set()
for a in repunits:
    for b in repunits:
        for c in repunits:
            s = a + b + c
            sums.add(s)

sorted_sums = sorted(sums)
print(sorted_sums[n-1])