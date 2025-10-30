def solve():
    repunits = [0] * 13
    repunit_value = 0
    for i in range(1, 13):
        repunit_value = repunit_value * 10 + 1
        repunits[i] = repunit_value
    
    sums = []
    for i1 in range(1, 13):
        for i2 in range(i1, 13):
            for i3 in range(i2, 13):
                current_sum = repunits[i1] + repunits[i2] + repunits[i3]
                sums.append(current_sum)
                
    sums.sort()
    
    n = int(input())
    print(sums[n-1])

if __name__ == '__main__':
    solve()