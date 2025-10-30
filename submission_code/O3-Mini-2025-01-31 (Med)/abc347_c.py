def main():
    import sys
    import bisect
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    A = int(next(it))
    B = int(next(it))
    T = A + B  # Total days in a week in AtCoder's kingdom.
    
    # Read days for each plan.
    D = [int(next(it)) for _ in range(N)]
    
    # We work modulo T (the week length). We want to choose an offset (unknown current day)
    # so that for each plan with day D_i, the day ((current_day_offset + D_i)-th day of week)
    # falls in a holiday; i.e. one of the first A days of a week.
    #
    # Reformulate: Let t be an unknown shift we can choose (0-indexed).
    # We require that for each plan i, 
    #      (D_i + t) mod T < A.
    # Define r_i = D_i mod T, then we need a t with 0 <= t < T such that for every i,
    #      (r_i + t) mod T < A.
    # In other words, if we cyclically shift the values r_i by t, they must lie in the range 0 to A-1.
    #
    # This is equivalent to asking: Is it possible to choose a contiguous segment of days of length A (the holidays)
    # on the circle of length T that covers all r_i?
    #
    # Let the sorted remainders be:
    mods = [d % T for d in D]
    mods.sort()
    
    # To deal with circle wrap-around easily, we duplicate the array with each element increased by T.
    ext = mods + [x + T for x in mods]
    
    # Now check for each candidate interval starting at ext[i] if it covers all N points.
    # The candidate holiday interval is [ext[i], ext[i] + A - 1].
    for i in range(N):
        target = ext[i] + A - 1
        # Find the first index in ext that has a value greater than target.
        j = bisect.bisect_right(ext, target)
        # Count how many points in ext, starting from i, are in the interval.
        if j - i >= N:
            sys.stdout.write("Yes")
            return
    sys.stdout.write("No")

if __name__ == '__main__':
    main()