def generate_repunits(limit):
    repunits = []
    num = 1
    while num <= limit:
        repunits.append(num)
        num = num * 10 + 1
    return repunits

N = int(input())

repunits = generate_repunits(10**12)
sums = []

for i in range(len(repunits)):
    for j in range(i, len(repunits)):
        for k in range(j, len(repunits)):
            sum_val = repunits[i] + repunits[j] + repunits[k]
            sums.append(sum_val)

sums = sorted(list(set(sums)))
print(sums[N-1])