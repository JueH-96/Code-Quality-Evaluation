def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Read the ball values.
    blacks = [int(next(it)) for _ in range(N)]
    whites = [int(next(it)) for _ in range(M)]
    
    # Explanation:
    # We want to choose some balls (possibly none) so that:
    #   (# chosen black balls) >= (# chosen white balls)
    # and maximize the total sum.
    #
    # Observation:
    #  • For blacks, any ball with value >= 0 is always beneficial.
    #    So we can always take all the blacks with nonnegative values.
    #  • For white balls, since a negative ball would lower the sum,
    #    it makes sense only to consider those with positive values.
    #
    # However, if we decide to pick x white balls (with x > 0),
    # we must have at least x black balls in our selection.
    # We already plan on taking all nonnegative blacks for free.
    # But if x exceeds the number of nonnegative blacks, then we are forced
    # to include some negative black balls. We choose them in the best order
    # (i.e. the negatives that are less damaging, sorted descending).
    
    # Partition blacks into "free" (nonnegative) and "forced" (negative).
    free_black = [b for b in blacks if b >= 0]
    forced_black = [b for b in blacks if b < 0]
    
    # Sorting:
    # For free_black order does not matter (we take all),
    # but for forced_black we sort in descending order,
    # so that if required we add the “least negative” ones first.
    free_black.sort(reverse=True)
    forced_black.sort(reverse=True)
    sum_free = sum(free_black)
    count_free = len(free_black)
    
    # Precompute prefix sums for forced_black.
    # prefix_forced[k] will be the sum of the first k forced (negative) blacks.
    prefix_forced = [0]
    for b in forced_black:
        prefix_forced.append(prefix_forced[-1] + b)
    
    # For white balls, we only consider positives.
    white_pos = [w for w in whites if w > 0]
    white_pos.sort(reverse=True)
    prefix_white = [0]
    for w in white_pos:
        prefix_white.append(prefix_white[-1] + w)
    
    # Now, let x be the number of white balls we choose.
    # Then, we must choose at least x black balls.
    # We already take all free_black (which are nonnegative).
    # If x > count_free then we need to force inclusion of (x - count_free) negative blacks.
    # In that case, candidate sum = sum_free + (sum of first (x - count_free) forced negatives)
    #                          + (sum of top x positive white balls).
    #
    # If x ≤ count_free then no extra negative black is necessary so that candidate sum =
    #                               sum_free + (sum of top x white positive balls).
    #
    # We iterate x from 0 (choosing no white ball) up to the maximum possible number
    # of white balls we can choose. Note also x cannot exceed N (the total number of blacks).
    best = sum_free  # x = 0: choose all free blacks, no white balls.
    max_white_choices = min(len(white_pos), N)
    for x in range(1, max_white_choices + 1):
        if x <= count_free:
            candidate = sum_free + prefix_white[x]
        else:
            forced_needed = x - count_free
            candidate = sum_free + prefix_forced[forced_needed] + prefix_white[x]
        if candidate > best:
            best = candidate
    # Also, choosing no ball (total = 0) is allowed.
    if best < 0:
        best = 0
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()