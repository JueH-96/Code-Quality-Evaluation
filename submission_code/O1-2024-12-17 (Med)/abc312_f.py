def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    # Separate items into three lists: pull-tab (P), regular (R), openers (O)
    P = []
    R = []
    O = []
    idx = 2
    for _ in range(N):
        t = int(data[idx]); x = int(data[idx+1])
        idx += 2
        if t == 0:
            P.append(x)   # pull-tab can
        elif t == 1:
            R.append(x)   # regular can
        else:
            O.append(x)   # can opener

    # Sort descending by value
    P.sort(reverse=True)
    R.sort(reverse=True)
    O.sort(reverse=True)

    # Prefix sums (so prefix[i] = sum of first i values in sorted-desc list)
    # Each array gets an extra 0 at the front for convenience
    Pprefix = [0]*(len(P)+1)
    for i in range(len(P)):
        Pprefix[i+1] = Pprefix[i] + P[i]
    Rprefix = [0]*(len(R)+1)
    for i in range(len(R)):
        Rprefix[i+1] = Rprefix[i] + R[i]
    Cprefix = [0]*(len(O)+1)
    for i in range(len(O)):
        Cprefix[i+1] = Cprefix[i] + O[i]

    Pcount = len(P)
    Rcount = len(R)
    Ocount = len(O)

    # f(T,i) gives total happiness if we pick i regular cans (i opened)
    # and T-i pull-tab cans (T-i) -- subject to them existing in sorted order.
    def f(T, i):
        return Rprefix[i] + Pprefix[T - i]

    # We'll iterate T = number of cans (pull-tabs + regular) we take.
    # Then o = M-T = number of openers.
    # capacity = sum of top o openers = Cprefix[o].
    # We can only open up to 'capacity' regular cans, and obviously i <= T, i <= Rcount,
    # and T-i <= Pcount.  We use a "sliding i" to find the best split R=i, P=T-i.

    res = 0
    i = 0  # This will track how many regular cans we open at each T.
           # We'll move it smoothly as T goes from 0..M.

    for T in range(M + 1):
        o = M - T  # how many openers we pick
        if o > Ocount:
            # Can't pick more openers than exist
            continue
        capacity = Cprefix[o]

        # Lower bound on i: T-i <= Pcount => i >= T - Pcount
        LB = max(0, T - Pcount)
        # Upper bound on i: i <= Rcount, i <= T, i <= capacity
        UB = min(Rcount, T, capacity)
        if LB > UB:
            continue

        # Clamp i into [LB, UB]
        if i < LB:
            i = LB
        elif i > UB:
            i = UB

        # "Hill-climb" up while it improves (because Rprefix is non-decreasing in i,
        # and Pprefix[T-i] is non-increasing in i, so the total is unimodal).
        while i < UB and f(T, i+1) >= f(T, i):
            i += 1
        # Also check if we might need to climb down (in case we were beyond the peak).
        while i > LB and f(T, i-1) >= f(T, i):
            i -= 1

        # Record best for this T
        val = f(T, i)
        if val > res:
            res = val

    print(res)

# Call main() to run the solution
if __name__ == "__main__":
    main()