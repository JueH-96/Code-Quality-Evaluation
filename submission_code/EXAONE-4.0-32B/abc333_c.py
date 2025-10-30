def main():
    repunits = []
    for L in range(1, 13):
        repunits.append((10**L - 1) // 9)
    
    sums_set = set()
    n_rep = len(repunits)
    for i in range(n_rep):
        for j in range(i, n_rep):
            for k in range(j, n_rep):
                total = repunits[i] + repunits[j] + repunits[k]
                sums_set.add(total)
                
    sorted_sums = sorted(sums_set)
    n_val = int(input().strip())
    print(sorted_sums[n_val-1])

if __name__ == '__main__':
    main()