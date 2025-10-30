def solve():
    n = int(input())
    
    repunits = []
    for i in range(1, 13):
        repunits.append(int("1" * i))
    
    sums = set()
    for i in range(len(repunits)):
        for j in range(len(repunits)):
            for k in range(len(repunits)):
                sums.add(repunits[i] + repunits[j] + repunits[k])
    
    sums = sorted(list(sums))
    print(sums[n-1])

solve()