def main():
    import sys
    import bisect

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    P = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    
    # Sort the side dish prices and compute prefix sums.
    B.sort()
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i + 1] = prefix[i] + B[i]
    
    total = 0
    # For each main dish price 'a', calculate the contribution from pairing it with all side dishes.
    for a in A:
        # We need to find how many side dishes make the sum strictly less than P:
        # i.e., a + b < P, which means b < P - a.
        threshold = P - a
        k = bisect.bisect_left(B, threshold)  # count of b values that are less than threshold
        
        # For those side dishes, the set meal price is a + b.
        # Total for these pairs is:
        #    k * a (for the main dish part) + sum of the corresponding b values.
        cost_with_actual_sum = a * k + prefix[k]
        
        # For the remaining side dishes, the price is capped at P.
        # There are M - k such side dishes.
        cost_with_cap = (M - k) * P
        
        total += cost_with_actual_sum + cost_with_cap

    sys.stdout.write(str(total))

if __name__ == "__main__":
    main()