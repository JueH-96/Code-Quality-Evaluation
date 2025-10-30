def solve():
    n = int(input())
    repunits = [0] * 12
    repunits[0] = 1
    for i in range(1, 12):
        repunits[i] = repunits[i-1] * 10 + 1
    
    sums = []
    for i in range(12):
        for j in range(12):
            for k in range(12):
                sums.append(repunits[i] + repunits[j] + repunits[k])
    
    sums.sort()
    print(sums[n-1])

solve()