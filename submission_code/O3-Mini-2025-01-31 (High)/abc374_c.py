def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    departments = list(map(int, data[1:]))
    total = sum(departments)

    # We use the "meet in the middle" strategy:
    # Split the list into two groups. For each group,
    # generate all possible subset sums.
    # Then, for each subset sum sA from the first group,
    # we use binary search on the sorted subset sums for the 
    # second group (sB) to find a sum such that sA+sB is as close
    # as possible to total/2. This minimizes the maximum
    # group sum in the partition: max(sA+sB, total-(sA+sB)).
    
    mid = n // 2
    groupA = departments[:mid]
    groupB = departments[mid:]
    
    sumsA = []
    sumsB = []
    
    # Generate all subset sums for group A recursively.
    def dfs_A(i, current):
        if i == len(groupA):
            sumsA.append(current)
        else:
            dfs_A(i + 1, current)
            dfs_A(i + 1, current + groupA[i])
    
    dfs_A(0, 0)
    
    # Generate all subset sums for group B recursively.
    def dfs_B(i, current):
        if i == len(groupB):
            sumsB.append(current)
        else:
            dfs_B(i + 1, current)
            dfs_B(i + 1, current + groupB[i])
    
    dfs_B(0, 0)
    
    # Sort sumsB so that we can quickly search for a candidate.
    sumsB.sort()
    
    best = float('inf')
    # Iterate over all subset sums from group A.
    # For each, find a candidate from group B such that the total sum s = sA + sB
    # is as close as possible to total/2.
    for sA in sumsA:
        target = total / 2 - sA
        pos = bisect.bisect_left(sumsB, target)
        if pos < len(sumsB):
            s = sA + sumsB[pos]
            candidate = max(s, total - s)
            best = min(best, candidate)
        if pos > 0:
            s = sA + sumsB[pos - 1]
            candidate = max(s, total - s)
            best = min(best, candidate)
            
    sys.stdout.write(str(int(best)))
    
if __name__ == '__main__':
    main()