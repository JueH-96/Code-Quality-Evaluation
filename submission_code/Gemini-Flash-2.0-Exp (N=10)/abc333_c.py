def solve():
    n = int(input())
    repunits = []
    for i in range(1, 12):
        repunits.append(int("1" * i))
    
    sums = []
    for i in range(len(repunits)):
        for j in range(len(repunits)):
            for k in range(len(repunits)):
                sums.append(repunits[i] + repunits[j] + repunits[k])
    
    sums = sorted(list(set(sums)))
    print(sums[n-1])

solve()