def main():
    import sys
    import bisect
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Sort a copy of A and create prefix sum array
    sorted_A = sorted(A)
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + sorted_A[i]
    total_sum = prefix[N]
    
    # for each element in A, we want the sum of all elements strictly greater than it.
    # Using binary search to find the first index in sorted_A that is > a
    res = []
    for val in A:
        idx = bisect.bisect_right(sorted_A, val)
        # Sum of numbers from index idx to end:
        sum_greater = total_sum - prefix[idx]
        res.append(str(sum_greater))
    
    sys.stdout.write(" ".join(res))
    
if __name__ == '__main__':
    main()