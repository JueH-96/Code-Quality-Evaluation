def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    healths = list(map(int, input_data[1:]))
    
    T = 0  # Global turn counter

    # For each enemy, we need to find the minimal number of consecutive attacks (x)
    # such that the cumulative damage is at least its health.
    # When attacking an enemy, we start the turns from T+1 to T+x.
    # On each turn, if turn T is a multiple of 3 the damage is 3,
    # and otherwise the damage is 1.
    # The damage on attack number j (which is global turn T+j) is:
    #     damage = 3 if (T+j) % 3 == 0 else 1.
    # We can note that for a block of x consecutive turns starting from T+1,
    # the count of turns that are multiples of 3 is:
    #     k = floor((T+x)/3) - floor(T/3)
    #
    # Hence total damage in these x turns is:
    #     damage_total = x + 2*k.
    #
    # We need to find the minimum x such that:
    #     x + 2 * ( (T+x)//3 - T//3 ) >= h
    #
    # Since h and T can be large, we use binary search for each enemy.
    
    for h in healths:
        lo = 0
        hi = h * 2  # A safe upper bound (since worst-case if no multiple-of-3 occurs, x==h)
        while lo < hi:
            mid = (lo + hi) // 2
            # Calculate count of multiples of 3 in the turns [T+1, T+mid]:
            count_mult3 = (T + mid) // 3 - T // 3
            total_damage = mid + 2 * count_mult3
            if total_damage >= h:
                hi = mid
            else:
                lo = mid + 1
        # lo is the minimal number of attacks needed to defeat this enemy.
        T += lo

    sys.stdout.write(str(T))
    
if __name__ == '__main__':
    main()