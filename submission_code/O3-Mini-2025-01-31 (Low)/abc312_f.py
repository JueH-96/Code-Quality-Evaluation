def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # We have three types:
    #   0: pull‐tab can – if obtained, immediately adds X to happiness.
    #   1: regular can – if obtained AND “opened” by a can opener you have, adds X to happiness.
    #   2: can opener – does not add happiness by itself but if you take it, you gain a capacity 
    #      of opening up to X regular cans.
    #
    # We must choose exactly M items.
    # Note that if you choose a regular can, you can only “realize” its happiness if you have 
    # enough can opener capacity among your chosen can opener items.
    #
    # Hence, if you choose c can openers (type 2) and (M-c) non‐opener items, then if you choose r of the (M-c)
    # items to be regular cans and the rest (M-c - r) to be pull‐tab cans, you
    # must have r not only <= (number of available regular cans) and (M-c - r) <= (available pull-tabs)
    # but also r must be no more than the total capacity available from the c chosen can openers.
    #
    # We organize our items into three lists.
    A = []  # pull-tab cans (type 0), each gives direct happiness.
    B = []  # regular cans (type 1), happiness only if you “open” them.
    C = []  # can openers (type 2), each gives a capacity of X (no direct happiness)

    for _ in range(N):
        t = int(next(it))
        x = int(next(it))
        if t == 0:
            A.append(x)
        elif t == 1:
            B.append(x)
        else:
            C.append(x)
    # Sort the items by descending order of happiness (or capacity for openers)
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)

    # Precompute prefix sums for A and B.
    # prefixA[k] = sum of the k highest pull-tab happiness values.
    prefixA = [0]
    for x in A:
        prefixA.append(prefixA[-1] + x)
    prefixB = [0]
    for x in B:
        prefixB.append(prefixB[-1] + x)
    # Likewise, for openers we want to know the maximum total capacity we can get by selecting
    # the top c openers.
    prefixC = [0]
    for x in C:
        prefixC.append(prefixC[-1] + x)

    # Let NA, NB, NC be counts.
    NA = len(A)
    NB = len(B)
    NC = len(C)

    # Our goal is to choose exactly M items.
    # We decide how many openers c (0 <= c <= min(M, NC)) to take.
    # Then, we must take the remaining non‐opener items (from A and B): let remaining = M - c.
    # In the non–opener group, suppose we take r regular cans (from B) and (remaining - r) pull–tabs
    # (from A). For the selection to be feasible we must have:
    #   • r <= NB  (we have that many regular cans)
    #   • (remaining - r) <= NA  (we have that many pull–tabs)
    #   • r <= (total capacity available from the chosen openers) = prefixC[c]
    #
    # In other words, r must lie in:
    #     r_min = max(0, remaining - NA)    (so that (remaining - r) <= NA)
    #     r_max = min(remaining, NB, prefixC[c])    (cannot take more than available and also must be <= opener capacity)
    #
    # For a given r (satisfying the constraints) the total happiness is:
    #      prefixB[r] + prefixA[remaining - r]
    #
    # Our aim is to choose, for each c, a value r in the interval [r_min, r_max] that maximizes this sum.
    #
    # An important observation is that because A and B are sorted in descending order,
    # the prefix sums are “concave” (more precisely, the differences decrease). Therefore,
    # for fixed remaining, the function
    #      f(r) = prefixB[r] + prefixA[remaining - r]
    # is unimodal (first increasing then decreasing) so that its maximum is achieved at one of the endpoints.
    #
    # Thus, we only need to check the two endpoints: r = r_min and r = r_max.
    ans = 0
    # iterate possible number c of openers to take:
    for c in range(0, min(M, NC) + 1):
        remaining = M - c  # these many items must come from (A U B)
        # First, if remaining is more than the total available non–opener items then skip.
        if remaining > NA + NB:
            continue
        # Total capacity available from chosen openers:
        cap = prefixC[c]
        # Determine the range for r (number of regular cans chosen).
        r_min = max(0, remaining - NA)  # so that pull-tabs chosen = remaining - r_min <= NA.
        r_max = min(remaining, NB, cap)
        if r_min > r_max:
            continue
        # Evaluate the candidate total happiness at the endpoints.
        cand1 = prefixB[r_min] + prefixA[remaining - r_min]
        cand2 = prefixB[r_max] + prefixA[remaining - r_max]
        candidate = cand1 if cand1 >= cand2 else cand2
        if candidate > ans:
            ans = candidate

    sys.stdout.write(str(ans))
    
if __name__ == "__main__":
    main()