def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    pos = 1

    # Helper: Count the exponent of prime p in x.
    def v_exponent(x, p):
        cnt = 0
        while x % p == 0:
            cnt += 1
            x //= p
        return cnt

    # Explanation:
    # We are given an array of integers and a number k.
    # Our task is to use operations (each operation adds 1 to an element) 
    # so that the product becomes divisible by k.
    # For k in {2, 3, 5} we need at least one factor of k in the product.
    # For k == 4, note that 4 = 2^2 so we need the product to have at least 2 factors of 2.
    #
    # The product’s exponent for prime p is the sum over each element's exponent.
    # We compute the baseline exponent sum. If it is already enough (base_sum >= req),
    # we output 0.
    #
    # Otherwise, for each element we determine the minimal number of increments (d)
    # needed to “upgrade” its contribution. For a given element a:
    #   baseline = v_exponent(a, p)
    #   We want to find the smallest d >= 1 such that 
    #       v_exponent(a + d, p) - baseline >= R,
    # where R is 1 or 2 depending on whether we want at least one extra factor 
    # (for k=2,3,5 or if one extra factor suffices) or two extra factors (for k=4).
    #
    # Because we can use upgrades from different elements, if we need extra 2 factors,
    # we can either:
    #   - Upgrade one element till its gain >= 2, or
    #   - Upgrade two different elements each yielding a gain >= 1.
    # We then take the option requiring the minimum total operations.
    
    out_lines = []
    for _ in range(t):
        n = int(data[pos]); pos += 1
        k = int(data[pos]); pos += 1
        arr = list(map(int, data[pos:pos+n]))
        pos += n

        # Determine p and required exponent.
        if k == 4:
            p = 2
            req = 2
        else:
            p = k  # For k = 2, 3, or 5 the prime factor is k itself.
            req = 1

        # Compute the total number of factors in the product (baseline sum).
        base_sum = 0
        baseVals = [0] * n
        for i, a in enumerate(arr):
            cnt = v_exponent(a, p)
            baseVals[i] = cnt
            base_sum += cnt

        # If the product is already divisible by k, no operation is needed.
        if base_sum >= req:
            out_lines.append("0")
            continue

        needed = req - base_sum  # This will be 1 (for k in {2,3,5} or if adding one factor suffices)
                               # or 2 (for k==4 when product lacks two factors of 2).

        # For each element, we compute the minimal extra cost needed to get an improvement.
        # best1: minimal cost d (>=1) so that the element’s additional exponent >= 1.
        # best2: minimal cost d (>=1) so that the element’s additional exponent >= 2.
        INF = 10**9
        option1 = []  # List of costs to achieve +1 improvement
        option2 = []  # List of costs to achieve +2 improvement
        # We only need to search a small range for d. (Numbers are small so try d from 1 to 15)
        for i, a in enumerate(arr):
            base_ = baseVals[i]
            best1 = INF
            best2 = INF
            for d in range(1, 16):
                new_val = a + d
                new_exp = v_exponent(new_val, p)
                gain = new_exp - base_
                if gain >= 1 and best1 == INF:
                    best1 = d
                if gain >= 2 and best2 == INF:
                    best2 = d
                if best1 != INF and best2 != INF:
                    break
            if best1 != INF:
                option1.append(best1)
            if best2 != INF:
                option2.append(best2)

        ans = INF
        # If we need 1 extra factor, answer is the minimal cost among those that can provide at least a +1 improvement.
        if needed == 1:
            if option1:
                ans = min(option1)
        # If we need 2 extra factors then we have two choices:
        # a) Upgrade one element to achieve an extra improvement of at least 2.
        # b) Upgrade two different elements, each giving at least an improvement of 1.
        elif needed == 2:
            if option2:
                ans = min(ans, min(option2))
            if len(option1) >= 2:
                option1.sort()
                candidate = option1[0] + option1[1]
                ans = min(ans, candidate)
        out_lines.append(str(ans))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()