def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])

    # We choose a maximum repunit length that is high enough so that when we add
    # three repunits (with repetition allowed) we have at least 333 distinct sums.
    # Using repunits of length 1 to 15 (i.e. numbers: 1, 11, 111, …) gives us 15^3 = 3375 sums,
    # which is more than enough.
    max_len = 15
    repunits = [0] * (max_len + 1)  # repunits[i] will store the repunit with i ones (i>=1)
    
    for i in range(1, max_len + 1):
        # A repunit with i ones is (10**i - 1) / 9.
        repunits[i] = (10**i - 1) // 9

    # Use a set to collect all possible sums of three repunits.
    sums_set = set()
    for i in range(1, max_len + 1):
        for j in range(1, max_len + 1):
            for k in range(1, max_len + 1):
                total = repunits[i] + repunits[j] + repunits[k]
                sums_set.add(total)
                
    # Sort the sums in ascending order.
    sorted_sums = sorted(sums_set)
    
    # Output the N-th smallest sum (1-indexed).
    if 0 <= N - 1 < len(sorted_sums):
        sys.stdout.write(str(sorted_sums[N - 1]))
    else:
        sys.stdout.write("0")

if __name__ == '__main__':
    main()