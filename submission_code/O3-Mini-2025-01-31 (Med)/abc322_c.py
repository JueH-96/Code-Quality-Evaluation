def main():
    import sys
    from bisect import bisect_left

    # Read input data
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return

    # Parse input values
    N = int(input_data[0])
    M = int(input_data[1])
    fireworks_days = list(map(int, input_data[2:2+M]))

    # For each day from 1 to N, use binary search to find the first fireworks day >= i.
    results = []
    for i in range(1, N + 1):
        index = bisect_left(fireworks_days, i)
        # The fireworks day at found index is guaranteed to be >= i.
        diff = fireworks_days[index] - i  # If fireworks on day i, diff is 0.
        results.append(str(diff))
    
    sys.stdout.write("
".join(results))


if __name__ == '__main__':
    main()