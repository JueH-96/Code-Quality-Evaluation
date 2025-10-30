def main():
    import sys
    import bisect

    # Read input
    data = sys.stdin.read().split()
    it = iter(data)

    # Read number of sleighs (N) and queries (Q)
    n = int(next(it))
    q = int(next(it))

    # Read the reindeer requirements for each sleigh and sort them
    R = [int(next(it)) for _ in range(n)]
    R.sort()

    # Build prefix sum array where prefix[i] is the sum of the smallest i requirements
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + R[i]

    # For each query, use binary search to find how many sleighs can be pulled
    answers = []
    for _ in range(q):
        X = int(next(it))
        # bisect_right finds the first index in 'prefix' where the prefix sum is greater than X.
        # Subtract 1 to get the maximum number of sleighs that satisfy the condition.
        idx = bisect.bisect_right(prefix, X) - 1
        answers.append(str(idx))
    
    # Output the results
    sys.stdout.write("
".join(answers))

if __name__ == '__main__':
    main()