def generate_repunits(count):
    repunits = []
    for i in range(1, count + 1):
        repunits.append(int('1' * i))
    return repunits

n_repunits = 12
repunits = generate_repunits(n_repunits)

sums = set()
for i in range(n_repunits):
    for j in range(i, n_repunits):
        for k in range(j, n_repunits):
            sums.add(repunits[i] + repunits[j] + repunits[k])

sorted_sums = sorted(list(sums))

N = int(input())
print(sorted_sums[N - 1])