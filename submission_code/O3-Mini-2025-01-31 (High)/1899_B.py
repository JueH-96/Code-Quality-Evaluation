def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    idx = 1
    res = []
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        # Read the weights of the n boxes
        a = list(map(int, data[idx:idx+n]))
        idx += n

        # Precompute prefix sums for fast group-sum calculation.
        # prefix[i] = a[0] + a[1] + ... + a[i-1] for i>=1, with prefix[0]=0.
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + a[i]

        best = 0
        # For each k (truck capacity) from 1 to n, we consider it only if n is divisible by k.
        # Since the loading option is possible only if we can partition the boxes into groups of exactly k.
        for k in range(1, n+1):
            if n % k != 0:
                continue
            m = n // k   # number of trucks
            # When there is only one truck, the difference is 0.
            if m == 1:
                diff = 0
            else:
                group_min = None
                group_max = None
                # Calculate the sum for each truck's load.
                for j in range(m):
                    # The j-th truck gets boxes from index j*k to (j+1)*k - 1.
                    cur_sum = prefix[(j+1)*k] - prefix[j*k]
                    if group_min is None:
                        group_min = cur_sum
                        group_max = cur_sum
                    else:
                        if cur_sum < group_min:
                            group_min = cur_sum
                        if cur_sum > group_max:
                            group_max = cur_sum
                diff = group_max - group_min
            if diff > best:
                best = diff
        res.append(str(best))
    sys.stdout.write("
".join(res))
    
if __name__ == '__main__':
    main()