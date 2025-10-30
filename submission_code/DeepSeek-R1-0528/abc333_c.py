def main():
    n = int(input().strip())
    repunits = []
    for i in range(1, 13):
        repunits.append((10**i - 1) // 9)
    
    sums_set = set()
    num_rep = len(repunits)
    for i in range(num_rep):
        for j in range(i, num_rep):
            for k in range(j, num_rep):
                total = repunits[i] + repunits[j] + repunits[k]
                sums_set.add(total)
                
    arr = sorted(sums_set)
    print(arr[n-1])

if __name__ == '__main__':
    main()