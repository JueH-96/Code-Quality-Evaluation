# YOUR CODE HERE
def solve():
    n = int(input())
    repunits = []
    for i in range(1, 13):
        rep = 0
        for j in range(i):
            rep = rep * 10 + 1
        repunits.append(rep)
    
    sums = []
    for i in range(13):
        for j in range(13):
            for k in range(13):
                sums.append(repunits[i] + repunits[j] + repunits[k])
    
    sums.sort()
    print(sums[n-1])

solve()