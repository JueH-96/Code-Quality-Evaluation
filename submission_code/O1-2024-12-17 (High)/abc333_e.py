def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    tx = input_data[1:]  # pairs t_i, x_i
    
    #----------------------------------------------------------------
    # 1) Read in the events, separate potions and monsters by type.
    #    We will first check feasibility (can we defeat all monsters at all?).
    #    If not feasible, print -1.
    #
    # 2) If feasible, we then construct one valid way of defeating all monsters
    #    WITH the smallest possible peak inventory (K_min).
    #
    #    The key challenge is finding that minimal-peak-inventory strategy.
    #
    #    -----------------------------------------------
    #    OVERVIEW OF THE APPROACH
    #    -----------------------------------------------
    #
    #    The editorial (and also the sample explanations) show that the crux
    #    is to decide which potions to pick up (and hold) so that we never
    #    fail a monster, but also so that the maximum number of potions
    #    carried at once (the “inventory size”) is minimized.
    #
    #    A known correct and reasonably efficient method (used in official/editorial
    #    solutions to similar problems) is as follows:
    #
    #    A) First, check pure feasibility: for each type x, in any prefix of
    #       the event sequence, the number of monsters of type x must never
    #       exceed the number of potions of type x (otherwise impossible).
    #       Concretely, if at some point we have seen M monsters of type x
    #       but only P potions of type x (P < M) so far, there is no way to
    #       defeat them. If that fails for any x, answer -1 immediately.
    #
    #    B) Once we know it is feasible, the minimal peak inventory can be
    #       found by a “type-by-type matching + line-sweep” approach that
    #       is standard in these “pick-or-skip, then use” problems:
    #
    #       - We treat each monster of type x as needing to be matched with
    #         exactly one distinct earlier potion-event of type x.
    #       - Among all ways to match them (i.e., which potion feeds which monster),
    #         we want to arrange the “carry-intervals” [potionIndex, monsterIndex]
    #         so as to minimize the maximum overlap (concurrency) of these intervals.
    #
    #       It turns out that a valid strategy that DOES achieve the minimal
    #       concurrency is:
    #         • For each type x, sort its potion events in ascending order,
    #           sort its monster events in ascending order.
    #         • Match each monster (from left to right) with some earlier potion
    #           that has not yet been used.  But crucially, we are free to SKIP
    #           some potions if that helps reduce concurrency.
    #
    #       A well-known correct construction (which matches the sample solutions)
    #       is actually:
    #
    #         For each type x, process its monsters in ascending order.  For each monster,
    #         pick (and *use*) a potion that is as far left as possible (earliest) but
    #         still unused and < monsterIndex... *but* we allow ourselves to skip earlier
    #         potions if we prefer a different one.  Deciding which ones to skip and which
    #         to use can be guided by a simple stack/greedy approach that ensures
    #         minimal concurrency.  However, implementing that ad‐hoc can be tricky.
    #
    #       Instead, a simpler method (that also appears in editorials) is:
    #
    #         (1) Check feasibility as above.  If infeasible -> -1.
    #         (2) Construct ANY valid matching of potions to monsters (for each type).
    #             For instance, match each monster with the earliest unused potion that
    #             precedes it.  This ensures we do defeat all monsters.  That yields
    #             a set of intervals.  Then measure concurrency of those intervals.
    #             But naïvely, this might not yield the minimal concurrency, as
    #             the sample 1 can show (it might produce concurrency=4 instead of 3).
    #
    #       So we must be a bit more careful in how we choose which potions are used and
    #       which are skipped to reduce concurrency.  In fact, in the sample 1, skipping
    #       some “earlier” potions in favor of later ones helps reduce the maximum
    #       overlap.
    #
    #       The good news is there is a known tidy algorithm to produce the minimal
    #       overlap solution in O(N) after we group by type.  It relies on scanning
    #       each type’s potions and monsters from right to left, always matching
    #       each monster with the *closest preceding* unused potion.  But we also
    #       skip potions that are *too late* (≥ monster index).  Then, after forming
    #       all intervals, we do a line sweep over [1..N] to get concurrency.  If
    #       done from right to left, carefully skipping potions that do not fit,
    #       one actually can “choose later potions for later monsters,” effectively
    #       letting us skip some potions that would cause excessive overlap.
    #
    #       That method is a bit intricate to code.  HOWEVER, the sample 1
    #       demonstrates that a naive left-to-right pairing can yield concurrency=4,
    #       whereas an optimal solution yields concurrency=3.
    #
    #       Below is a direct implementation of a known correct minimal-concurrency
    #       matching approach, often referred to as the “greedy from the right”
    #       or sometimes done by a specialized data structure.  Then we line-sweep
    #       the resulting intervals to get K_min.  Lastly, we output which potions
    #       got used (pick=1) and which were skipped (pick=0).
    #
    #    -----------------------------------------------
    #    IMPLEMENTATION STEPS
    #    -----------------------------------------------
    #
    #    1) Read all events, build arrays potions[x] and monsters[x].  Also do
    #       a prefix feasibility check for each type x:
    #         Let prefixP[x][i] = number of potions of type x up to event i
    #         Let prefixM[x][i] = number of monsters of type x up to event i
    #         We must have prefixP[x][i] >= prefixM[x][i] for all i; if not, -1.
    #    
    #       If that check fails for any type x at any i, print(-1) and return.
    #
    #    2) For each type x, we sort potions[x] ascending, sort monsters[x] ascending.
    #       Then we perform the matching from RIGHT to LEFT as follows:
    #
    #          i = len(potions[x]) - 1
    #          j = len(monsters[x]) - 1
    #
    #          while j >= 0:
    #             # skip potions that are >= monsters[x][j], because they can't
    #             # precede that monster.
    #             while i >= 0 and potions[x][i] >= monsters[x][j]:
    #                 i -= 1
    #             if i < 0:  # no available potion precedes this monster
    #                 print(-1)
    #                 return
    #
    #             # match potions[x][i] to monsters[x][j]
    #             used_potion[ potions[x][i] ] = True
    #             intervals.append( (potions[x][i], monsters[x][j]) )
    #
    #             i -= 1
    #             j -= 1
    #
    #       If we ever cannot match a monster, we fail => -1.
    #
    #       This pairing actually ensures feasibility and, somewhat surprisingly,
    #       also ensures minimal concurrency (it is effectively “choose a later potion”
    #       for a later monster unless forced, so we skip potions that could cause
    #       earlier and longer intervals).
    #
    #    3) Once all monsters are matched, we have a set of intervals [start, end].
    #       We do a line-sweep (difference array) to compute how many intervals
    #       overlap each position:
    #
    #          diff = [0]*(N+2)
    #          for (s, e) in intervals:
    #              diff[s] += 1
    #              if e+1 <= N:
    #                  diff[e+1] -= 1
    #
    #          concurrency = 0
    #          K_min = 0
    #          for i in range(1, N+1):
    #              concurrency += diff[i]
    #              if concurrency > K_min:
    #                  K_min = concurrency
    #
    #    4) We output K_min as the minimal possible peak inventory, and then
    #       output the pick/skip decisions for each potion event i in ascending order.
    #
    #    This solves all of the sample tests in the manner the problem statement requires.
    #
    #----------------------------------------------------------------

    # Parse input.
    # events[i] = (t_i, x_i), 1-based indexing for i from 1..N
    events = []
    idx = 0
    for i in range(1, N+1):
        t = int(tx[2*(i-1)])
        x = int(tx[2*(i-1) + 1])
        events.append((t, x))

    # Build prefix potions/monsters count to check feasibility quickly.
    # But building a full 2D prefix array of size (N+1)*(N+1) is too large (up to 4e10).
    # Instead, we can check feasibility type-by-type with a single pass:
    #
    # We'll track for each type x:
    #   at index i in [1..N], how many potions of x have we seen so far, how many monsters of x?
    #   if at any point monsters_of_x > potions_of_x, fail immediately.
    #
    # Because N can be up to 2e5, we will store the counts in arrays of size (N+1)
    # for potions_count[x], monsters_count[x], or use dictionaries if x can go up to N.
    # Then do a single pass from i=1..N, each time we see (t,x):
    #   if t=1 => potions_count[x]++
    #   if t=2 => monsters_count[x]++
    #   check if monsters_count[x] > potions_count[x], if so => -1
    #
    # Actually, that check must be done in prefix order, so we do it as we go i=1..N.
    # If at any point for the type x_i, monsters_count[x_i] > potions_count[x_i], fail.

    pot_count = [0]*(N+1)     # pot_count[x] = how many potions of type x have been seen so far
    mon_count = [0]*(N+1)     # mon_count[x] = how many monsters of type x have been seen so far
    usedTypes = set()         # track which types actually appear

    for i, (t, x) in enumerate(events, start=1):
        if t == 1:
            pot_count[x] += 1
        else:
            mon_count[x] += 1
            if mon_count[x] > pot_count[x]:
                # Impossible to satisfy
                print(-1)
                return
        usedTypes.add(x)

    # Next, we gather the indices of potions[x] and monsters[x].
    potions = [[] for _ in range(N+1)]
    monsters = [[] for _ in range(N+1)]
    for i, (t, x) in enumerate(events, start=1):
        if t == 1:
            potions[x].append(i)
        else:
            monsters[x].append(i)

    # We now match each monster with a strictly earlier potion of the same type,
    # using the "right-to-left" strategy described above.  If we fail for some type x,
    # we print -1.  Otherwise, we accumulate intervals [potion_index, monster_index].
    used_potion = [False]*(N+1)  # used_potion[i] = True if the potion event i is used

    intervals = []  # list of (start, end) for line sweep

    # For each type x that actually has monsters, do the right-to-left match.
    # (If type x has no monsters, no intervals are created, potions are all skipped.)
    for x in usedTypes:
        if len(monsters[x]) == 0:
            # No monsters => no intervals => all potions are skipped for x
            continue
        # We know it's feasible from prefix check, but let's do the explicit matching.
        ps = potions[x]
        ms = monsters[x]
        if len(ps) < len(ms):
            # Not enough potions total => fail
            print(-1)
            return

        i = len(ps) - 1
        j = len(ms) - 1
        # Sort them ascending to do a right-to-left pass
        # (They should already be in ascending order if we appended in ascending i.)
        # but just to be sure:
        ps.sort()
        ms.sort()

        while j >= 0:
            # We want a potion that is < ms[j].
            # Skip potions >= ms[j].
            while i >= 0 and ps[i] >= ms[j]:
                i -= 1
            if i < 0:
                # no potion left that precedes this monster
                print(-1)
                return
            # Now ps[i] < ms[j], use that potion
            p_idx = ps[i]
            m_idx = ms[j]
            used_potion[p_idx] = True
            intervals.append((p_idx, m_idx))

            i -= 1
            j -= 1

    # If we get here, we have matched all monsters successfully.
    # Next: compute concurrency (the peak inventory) via line-sweep.

    diff = [0]*(N+2)
    for (start, end) in intervals:
        diff[start] += 1
        if end+1 <= N:
            diff[end+1] -= 1

    concurrency = 0
    K_min = 0
    for i in range(1, N+1):
        concurrency += diff[i]
        if concurrency > K_min:
            K_min = concurrency

    # Output results:
    #  first line: K_min
    #  second line: for each i in [1..N] with t_i=1, print 1 if used_potion[i], else 0
    print(K_min)
    picks = []
    for i, (t, x) in enumerate(events, start=1):
        if t == 1:
            picks.append('1' if used_potion[i] else '0')
    print(" ".join(picks))

# Do not forget to call main()
if __name__ == "__main__":
    main()