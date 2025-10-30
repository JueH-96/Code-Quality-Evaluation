def solve():
    import sys
    N = int(sys.stdin.readline().strip())
    
    # Generate repunits up to 40 digits (more than enough for N ≤ 333)
    repunits = []
    current = 0
    for _ in range(40):
        current = current * 10 + 1
        repunits.append(current)
    
    # Generate all possible sums of exactly three repunits
    sums_of_three = set()
    for i in range(40):
        for j in range(40):
            for k in range(40):
                sums_of_three.add(repunits[i] + repunits[j] + repunits[k])
    
    # Sort and output the N-th smallest (1-based index)
    sorted_sums = sorted(sums_of_three)
    print(sorted_sums[N-1])

def main():
    solve()

if __name__ == "__main__":
    main()