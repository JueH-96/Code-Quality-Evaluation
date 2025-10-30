def main():
    import sys, math
    data = sys.stdin.read().split()
    t = int(data[0])
    pos = 1
    results = []
    for _ in range(t):
        n = int(data[pos])
        pos += 1
        a = list(map(int, data[pos: pos+n]))
        pos += n

        # Build prefix sums with prefix[0] = 0.
        prefix = [0] * (n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + a[i]
        
        # We only consider valid k that allow dividing n exactly, i.e., divisors of n.
        # Get all divisors of n.
        divisors = []
        for x in range(1, int(math.isqrt(n)) + 1):
            if n % x == 0:
                divisors.append(x)
                if x * x != n:
                    divisors.append(n // x)

        max_diff = 0
        # For each valid k, compute the sums for each truck.
        for k in divisors:
            trucks = n // k
            current_min = None
            current_max = None
            for j in range(trucks):
                # Sum for j-th truck using prefix sums.
                s = prefix[(j+1)*k] - prefix[j*k]
                if current_min is None or s < current_min:
                    current_min = s
                if current_max is None or s > current_max:
                    current_max = s
            diff = current_max - current_min
            if diff > max_diff:
                max_diff = diff

        results.append(str(max_diff))
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()