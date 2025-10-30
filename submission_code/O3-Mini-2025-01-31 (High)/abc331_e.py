def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    # Read N, M, L
    N = int(next(it))
    M = int(next(it))
    L = int(next(it))
    
    # Read main dish prices and side dish prices.
    a = [int(next(it)) for _ in range(N)]
    b = [int(next(it)) for _ in range(M)]
    
    # Build a list of sets for forbidden side dishes for each main dish.
    # forbidden[i] will be the set of side dish indices (0-indexed) that are disallowed with main dish i.
    forbidden = [set() for _ in range(N)]
    for _ in range(L):
        c = int(next(it))
        d = int(next(it))
        forbidden[c - 1].add(d - 1)
    
    # Precompute sorted_sides: a list of side dish indices sorted in descending order by price.
    sorted_sides = sorted(range(M), key=lambda j: b[j], reverse=True)
    # The best possible side dish (price) if there is no restriction.
    max_side_price = b[sorted_sides[0]] if M > 0 else 0
    
    best = 0
    # For each main dish, try to choose the best allowed side dish.
    for i in range(N):
        main_price = a[i]
        # If this main dish has no forbidden side dishes, then we can always choose the side dish with max price.
        if not forbidden[i]:
            candidate_side = max_side_price
        else:
            candidate_side = None
            # Otherwise, scan the side dishes (already sorted from highest to lowest)
            # until we find a side dish not in the forbidden set.
            for j in sorted_sides:
                if j not in forbidden[i]:
                    candidate_side = b[j]
                    break
        # If an allowed side dish was found, update the best obtained sum.
        if candidate_side is not None:
            total = main_price + candidate_side
            if total > best:
                best = total
    
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()