def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    arr = [int(next(it)) for _ in range(n)]
    
    # Determine the maximum value in the array.
    max_val = max(arr)
    
    # Build frequency array for values 1...max_val.
    freq = [0] * (max_val + 1)
    for a in arr:
        freq[a] += 1
        
    # Build prefix sums to quickly compute frequency sums in intervals.
    prefix = [0] * (max_val + 1)
    cum = 0
    for i in range(max_val + 1):
        cum += freq[i]
        prefix[i] = cum

    total = 0

    # First, account for pairs where both numbers are the same.
    # For a pair (a,a), the contribution is floor(a/a) = 1.
    for a in range(1, max_val + 1):
        if freq[a] > 1:
            total += freq[a] * (freq[a] - 1) // 2

    # Next, account for pairs with distinct values.
    # We choose a as the smaller value and b as the larger value.
    # For a fixed a, consider multiples k so that for b where floor(b / a) = k,
    # b must lie in the interval:
    #    [k*a, (k+1)*a - 1]
    # For k = 1, we must ensure b > a (to avoid counting pairs with same values).
    for a in range(1, max_val + 1):
        if freq[a] == 0:
            continue
        # For each multiplier k where there could be some numbers with that quotient.
        max_k = max_val // a  # largest possible k such that k*a <= max_val.
        for k in range(1, max_k + 1):
            if k == 1:
                # When k is 1, b should be in [a, 2*a - 1] but excluding b == a.
                L = a + 1
                R = 2 * a - 1
            else:
                L = k * a
                R = (k + 1) * a - 1

            if L > max_val:  # No b can be >= L if L exceeds max_val.
                break
            if R > max_val:
                R = max_val

            if L > R:
                continue

            # Use prefix array to calculate the total frequency of values in [L, R].
            count_b = prefix[R] - prefix[L - 1]
            if count_b:
                total += k * freq[a] * count_b

    sys.stdout.write(str(total))


if __name__ == '__main__':
    main()