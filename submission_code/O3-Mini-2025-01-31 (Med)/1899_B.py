def main():
    import sys, math
    data = sys.stdin.buffer.read().split()
    t = int(data[0])
    idx = 1
    res = []
    
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n

        # Build prefix sums: prefix[i] = sum of first i boxes
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]

        # Only those k which evenly divide n are valid.
        divisors = []
        root = int(math.isqrt(n))
        for d in range(1, root + 1):
            if n % d == 0:
                divisors.append(d)
                if d * d != n:
                    divisors.append(n // d)

        # For each allowed truck capacity k, compute the truck group sums and then the difference.
        best = 0
        for k in divisors:
            m = n // k  # number of trucks
            group_min = None
            group_max = None
            for i in range(m):
                s = prefix[(i + 1) * k] - prefix[i * k]
                if group_min is None or s < group_min:
                    group_min = s
                if group_max is None or s > group_max:
                    group_max = s
            # The maximum absolute difference among truck totals for this k
            diff = group_max - group_min
            if diff > best:
                best = diff

        res.append(str(best))
    
    sys.stdout.write("
".join(res))

if __name__ == '__main__':
    main()