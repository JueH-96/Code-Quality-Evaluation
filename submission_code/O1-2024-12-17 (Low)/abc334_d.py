def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    R = list(map(int, input_data[2:2+N]))
    queries = list(map(int, input_data[2+N:]))

    # Sort the list of required reindeer
    R.sort()
    
    # Build prefix sums of sorted R
    prefix_sum = [0] * (N+1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + R[i]

    import bisect

    # For each query, perform a binary search on prefix_sum
    idx = 0
    start = 0
    output = []
    for x in queries:
        # We want the largest k such that prefix_sum[k] <= x
        # bisect_right finds position to the right, so subtract 1
        # But we can directly do:
        # k = bisect.bisect_right(prefix_sum, x) - 1
        # This k is the maximum index where prefix_sum[k] <= x
        k = bisect.bisect_right(prefix_sum, x) - 1
        
        # k might go beyond N (if x is large), so min with N
        k = min(k, N)
        output.append(str(k))

    print("
".join(output))

# Do not forget to call main
if __name__ == "__main__":
    main()