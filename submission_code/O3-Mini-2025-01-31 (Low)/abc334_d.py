def main():
    import sys, bisect
    
    input_data = sys.stdin.read().split()
    it = iter(input_data)
    n = int(next(it))
    q = int(next(it))
    
    reindeer_requirements = [int(next(it)) for _ in range(n)]
    # Sort reindeer requirements to minimize the total reindeer required for any selected number of sleighs.
    reindeer_requirements.sort()
    
    # Build prefix sums of the sorted requirements.
    prefix_sums = [0] * (n + 1)
    for i in range(n):
        prefix_sums[i+1] = prefix_sums[i] + reindeer_requirements[i]
    
    results = []
    for _ in range(q):
        X = int(next(it))
        # Want maximum m such that prefix_sums[m] <= X.
        # Use binary search on the prefix_sums array.
        # bisect_right returns the insertion point to the right of X, so we subtract 1.
        m = bisect.bisect_right(prefix_sums, X) - 1
        results.append(str(m))
    
    # Output each result on a new line.
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()