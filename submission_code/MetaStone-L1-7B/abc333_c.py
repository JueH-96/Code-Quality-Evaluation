n = int(input())
repunits = []
for k in range(1, 21):
    repunits.append(int('1' * k))

sums = set()
for a in repunits:
    for b in repunits:
        for c in repunits:
            sums.add(a + b + c)

sorted_sums = sorted(sums)
print(sorted_sums[n-1])