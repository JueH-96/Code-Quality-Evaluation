def main():
    import sys, bisect

    # Read input from stdin
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    K = list(map(int, data[1:]))

    total = sum(K)
    
    # We generate all possible subset sums.
    # Since N is at most 20, the total number of subsets is 2^N (up to about 1e6).
    subset_sums = [0]
    for num in K:
        # For every existing subset sum, add the current number to form new subset sums
        new_sums = [s + num for s in subset_sums]
        subset_sums += new_sums

    # Sort the list to enable binary search for the optimal subset sum near total//2.
    subset_sums.sort()
    
    # Our goal is to partition departments so that the load of each group is as balanced as possible.
    # Let candidate subset have sum s, then the other group has sum total-s.
    # We want to minimize the maximum of these two sums.
    # The ideal case is when s is as close to total/2 as possible.
    target = total // 2

    # Using binary search, find index where we could insert the target.
    idx = bisect.bisect_right(subset_sums, target)
    best = total  # Initialize best answer with a value not greater than total.
    
    # Check the candidate just below and just above the target.
    # These are the two most promising subset sums for a balanced partition.
    candidates = []
    if idx > 0:
        candidates.append(subset_sums[idx-1])
    if idx < len(subset_sums):
        candidates.append(subset_sums[idx])
    
    # Evaluate each candidate
    for s in candidates:
        cand = max(s, total - s)
        if cand < best:
            best = cand

    # Output the answer to stdout.
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()