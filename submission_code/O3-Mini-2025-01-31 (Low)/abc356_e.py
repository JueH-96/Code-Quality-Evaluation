def main():
    import sys
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    max_val = max(A)
    # Create a frequency array for values from 0 to max_val.
    f = [0] * (max_val + 1)
    for a in A:
        f[a] += 1
        
    # Build a prefix sum array to quickly compute frequencies in an interval.
    prefix = [0] * (max_val + 1)
    current = 0
    for i in range(max_val + 1):
        current += f[i]
        prefix[i] = current

    # Helper function: return the total count in f[low...high].
    def count_in_range(low, high):
        if low > high or low > max_val:
            return 0
        if low <= 0:
            return prefix[high]
        return prefix[high] - prefix[low - 1]

    ans = 0
    # First, take into account pairs with equal numbers.
    # For a pair (x,x), floor(x/x)=1. So each such pair contributes 1.
    for x in range(1, max_val + 1):
        if f[x] > 1:
            ans += f[x] * (f[x] - 1) // 2

    # Now, consider pairs (A_i, A_j) with A_i < A_j.
    # For a fixed smaller value x and a larger value y, with y in [x+1, max_val],
    # the term floor(y/x) equals k when y ∈ [k*x, (k+1)*x - 1].
    # We iterate over all multiples k for each x.
    for x in range(1, max_val + 1):
        if f[x] == 0:
            continue
        # k starts from 1. For k==1, note that we already counted pairs where y equals x,
        # so for y we must consider only values > x.
        k = 1
        while x * k <= max_val:
            L = x * k
            R = min(max_val, x * (k + 1) - 1)
            if k == 1:
                # Exclude y == x already counted.
                L = x + 1
            if L > R:
                k += 1
                continue
            cnt = count_in_range(L, R)
            # f[x] is the frequency for the smaller element.
            # Each pair (x, y) with y in the interval contributes k.
            ans += f[x] * cnt * k
            k += 1

    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()