def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parsing input
    N = int(input_data[0])
    X = list(map(int, input_data[1:N+1]))
    Q = int(input_data[N+1])
    T = []
    G = []
    idx = N+2
    for _ in range(Q):
        t_i = int(input_data[idx]); g_i = int(input_data[idx+1])
        T.append(t_i); G.append(g_i)
        idx += 2

    # ----------------------------------------------------------------
    # PROBLEM RECAP
    #
    # We have N people on a 1D infinite integer line, initially at
    # positions X[0], X[1], ..., X[N-1], where X is strictly ascending.
    # Person i (1-based) is at X[i-1].  We have Q tasks in sequence;
    # the k-th task says "person T[k] must end at position G[k]."
    #
    # At the end of each of the Q tasks (in order), we must have
    # all persons in distinct integer positions, still in strictly
    # increasing order by their person-index (i.e. person1 < person2 < ... < personN),
    # and person T[k] exactly at G[k].  We want to minimize the total
    # sum of all 1-step moves used across all Q tasks.
    #
    # Key insight (which is standard in such "no-passing" 1D problems):
    # The minimal‐cost way (in total) never has persons cross each other.
    # Hence the left‐to‐right order of persons is invariant, so after
    # each step they remain in ascending order p_1 < p_2 < ... < p_N.
    #
    # Furthermore, one can re‐interpret each p_i via f_i = p_i - i,
    # which must remain a nondecreasing sequence f_1 <= f_2 <= ... <= f_N
    # (because p_i < p_{i+1}  <=>  (p_i - i) <= (p_{i+1} - (i+1)) ).
    #
    # An update "person T goes to G" means p_T = G => f_T = G - T.
    # Then we must retain f_1 <= f_2 <= ... <= f_T <= ... <= f_N,
    # adjusting f_i minimally from the previous step's values, to keep
    # the sequence nondecreasing.  The cost of each step is
    #   sum_{i=1..N} | f_i(new) - f_i(old) |,
    # because moving from p_i(old) to p_i(new) is the same as changing
    # f_i(old) to f_i(new) (the i offset cancels in the absolute difference).
    #
    # The update is simple "clamping":
    #   If we set f_T = v, where v = G - T,
    #   - if v > old f_T, we 'push' to the right: all consecutive f_i
    #     that are < v must be raised up to v, up to the first that is >= v.
    #   - if v < old f_T, we 'push' to the left: all consecutive f_i
    #     that are > v must be lowered to v, back to the first that is <= v.
    #
    # We sum up all these absolute differences as cost.  Initially,
    # f_i = X[i-1] - i for i=1..N.
    #
    # Data-Structure:
    # We'll keep f as a nondecreasing array in "interval" form:
    #   intervals = [(start_index, end_index, value), ...]
    # sorted by start_index, and the values are in nondecreasing order.
    #
    # For each update:
    #   1) locate the interval containing T_k (via binary search).
    #   2) set oldVal = that interval's value.  If oldVal == v, cost=0, done.
    #   3) if v > oldVal => "raise to the right":
    #        - possibly split the interval at T_k if T_k is not at boundary.
    #        - from that piece, move right merging all intervals with value <= v,
    #          raising them to v, accumulating cost = sum_{indices}(v - old_value).
    #        - stop when we reach an interval with value > v (which is fine),
    #          or run out of intervals.  Also merge with any interval that already
    #          has value == v (since they can be combined).
    #      if v < oldVal => similarly "lower to the left."
    #
    # The total cost is accumulated.  In the end we print it.
    #
    # This matches exactly the examples given in the problem statement.
    #
    # Complexity: Each update is O(log(#intervals) + merges).  Each interval is
    # merged or split at most once or twice per update.  So total merges/splits
    # over Q steps is O(N+Q).  The total complexity is O((N+Q) log(N+Q)) which
    # should be acceptable for N,Q up to 2e5 in a fast language.  In Python,
    # it may be borderline but usually can pass if implemented efficiently.
    #
    # Let's implement now.
    # ----------------------------------------------------------------

    sys.setrecursionlimit(10**7)

    # Build initial f array in 1-based indexing:
    # f[i] = X[i-1] - i
    f = [0]*(N+1)
    for i in range(1, N+1):
        f[i] = X[i-1] - i

    # We'll convert f into intervals of constant value.  (Because f is nondecreasing,
    # consecutive indices either have the same value or increase.)
    # intervals: list of (start, end, value)
    intervals = []
    start = 1
    curval = f[1]
    for i in range(2, N+1):
        if f[i] != curval:
            # close the old interval
            intervals.append((start, i-1, curval))
            start = i
            curval = f[i]
    # close the last
    intervals.append((start, N, curval))

    # A helper to do a binary-search for the interval containing index pos
    # intervals are disjoint, sorted by (start, end) with no overlap.
    # We'll return the index i in [0..len(intervals)-1].
    def find_interval(pos):
        # standard binary search on intervals' start/end
        lo, hi = 0, len(intervals)-1
        while lo <= hi:
            mid = (lo+hi)//2
            s, e, v = intervals[mid]
            if s <= pos <= e:
                return mid
            elif pos < s:
                hi = mid-1
            else:
                lo = mid+1
        return -1  # should never happen if pos is in [1..N]

    # We'll maintain a running total cost in a (possibly large) integer
    total_cost = 0

    # We'll define some helper functions to manipulate intervals.

    # (1) split_interval(i, pos): if intervals[i] covers [s..e], and s < pos <= e,
    #     we split into up to two intervals: [s..pos-1], [pos..e], both with the same value.
    #     We return the index of the interval that starts at pos after split.
    def split_interval(i, pos):
        # i is index in intervals
        s, e, val = intervals[i]
        if s == pos:
            return i  # no split needed, the interval already starts at pos
        # else we split
        # first part: (s, pos-1, val)
        # second part: (pos, e, val)
        new_int = (pos, e, val)
        intervals[i] = (s, pos-1, val)
        intervals.insert(i+1, new_int)
        return i+1

    # (2) merge intervals [i..j] into one interval with value = newVal.
    #     compute the cost of raising or lowering that entire block from oldVal to newVal.
    #     Actually, each interval might have a different oldVal, so we must sum up properly.
    #     We'll do it in a loop, summing cost as we go, then unify them.
    def merge_right(i, j, newVal):
        # intervals from i..j are contiguous in index,
        # and we will unify them into one big interval with value = newVal.
        nonlocal total_cost
        # compute cost by summing (newVal - oldVal)*length if newVal>oldVal, or
        # (oldVal - newVal)*length if newVal<oldVal, but we'll just do abs difference.
        # Actually we know the direction from the calling code, but let's just do the
        # difference carefully so we can reuse.
        cost_sum = 0
        start_ = intervals[i][0]
        end_ = intervals[j][1]
        for idx in range(i, j+1):
            s, e, oldVal = intervals[idx]
            length = e - s + 1
            cost_sum += abs(newVal - oldVal)*length
        total_cost += cost_sum
        # unify into one interval
        intervals[i] = (start_, end_, newVal)
        # remove i+1..j
        # slice deletion in python can be O(k), but j-i can be large. We rely on total merges
        # being O(N+Q). So it's acceptable amortized.
        del intervals[i+1:j+1]

    # We'll do the "raise to the right" operation: starting from interval i0,
    # we unify all consecutive intervals whose val <= newVal (since f must be nondecreasing).
    # Actually we unify as long as intervals[next].val <= newVal, because if next.val > newVal,
    # that is still monotonic: next.val >= newVal is fine, so we must STOP if next.val > newVal?
    # Actually if next.val == newVal, we can unify it (no cost).
    # If next.val < newVal, we raise it. If next.val > newVal, that is fine, no unify needed,
    # we stop. Because we only "push" intervals that are strictly < newVal or equal (since they
    # can be merged).
    #
    # In effect, we keep going while intervals[j].val <= newVal.  Then we do merge_right(i0, j, newVal).
    def raise_right(i0, newVal, pos):
        # pos is the actual index T_k where we do the clamp, used for partial splitting etc.
        # We'll do a while loop from j=i0 onward.
        j = i0
        nI = len(intervals)
        val_j = intervals[j][2]
        # we already know val_j < newVal or val_j == newVal for raising scenario
        # but let's unify as far as next intervals have val <= newVal:
        while True:
            if j+1 >= len(intervals):
                # unify everything from i0..(end)
                merge_right(i0, j, newVal)
                break
            # check next interval
            if intervals[j+1][2] <= newVal:
                # continue
                j += 1
            else:
                # unify i0..j
                merge_right(i0, j, newVal)
                # check if next interval has exactly newVal => unify that too
                # but let's see intervals[i0] might have changed
                if j < len(intervals) and j+1 < len(intervals):
                    # j might no longer be valid if merges shrank the list
                    # so let's recast j in case merges removed intervals
                    # but we have just done merge_right which modifies intervals up to j
                    # so intervals[i0] is now the merged one; the next interval is i0+1
                    if i0+1 < len(intervals) and intervals[i0+1][2] == newVal:
                        merge_right(i0, i0+1, newVal)
                break

    # Similarly, "lower to the left": from interval i0, unify intervals to the left
    # that have val >= newVal (since f must be nondecreasing, left side can't exceed right side).
    # We keep going while intervals[i0-1].val >= newVal.
    def lower_left(i0, newVal, pos):
        # go left
        j = i0
        while True:
            if j-1 < 0:
                # unify everything from j..i0
                merge_right(j, i0, newVal)  # but j>i0 doesn't make sense, so we interpret i0..j
                break
            if intervals[j-1][2] >= newVal:
                j -= 1
            else:
                merge_right(j, i0, newVal)
                # possibly unify j-1 if it has val == newVal
                # after merging i0..j into a single interval at index j
                # we recalc j
                # Actually we should unify in ascending index order, so we want j < i0
                # let's fix: we do merge_right(min(j, i0), max(j, i0), newVal).
                #   j <= i0 by construction if we are going left
                merge_right(j, i0, newVal)
                # the newly merged interval is now at index j
                if j-1 >= 0 and intervals[j-1][2] == newVal:
                    merge_right(j-1, j, newVal)
                break

    # But the above approach reuses merge_right in a slightly confusing manner.
    # Actually "merge_right(l, r, newVal)" just merges intervals in the range [l..r].
    # For lowering to the left we want to unify [j..i0]. So j <= i0. So that is okay,
    # as long as we pass them in ascending order. We can do merge_right(j, i0, newVal).
    #
    # Let's do a simpler approach: we'll gather the "left range" first, then do one big merge.
    # Implementation note: to keep it symmetrical, we'll just do an explicit approach.

    def do_raise(i, pos, v):
        """
        We have intervals[i] containing pos, oldVal < v.
        We'll raise intervals to the right while their value <= v.
        Then unify them.  That ensures f remains nondecreasing.
        """
        # first unify i.. ??? as in the function raise_right
        # but we must re-check i because we might split interval i if pos is not the start.
        s, e, oldVal = intervals[i]
        # split if needed
        if s < pos:
            # split off [s..pos-1]
            i = split_interval(i, pos)  # i now points to [pos..e]
            s, e, oldVal = intervals[i]

        # now intervals[i] = (pos, e, oldVal) with oldVal < v
        # we do the raise_right
        raise_right(i, v, pos)

    def do_lower(i, pos, v):
        """
        intervals[i] containing pos, oldVal > v
        We'll lower intervals to the left while their value >= v.
        Then unify them.  That ensures nondecreasing property.
        """
        s, e, oldVal = intervals[i]
        if s < pos:
            # we might want to split first so that pos is the end of the new interval? Actually if pos < e
            # we want the interval containing pos to start exactly at s, so if pos < e, we do an extra split?
            # Actually we want to isolate the portion [pos..e], so we do:
            i = split_interval(i, pos)
            s, e, oldVal = intervals[i]
            # now intervals[i] is [pos..e]
            # but we want pos exactly to be that interval's start, that's correct.

        # Actually if pos> s, it's the same logic, let's see:
        # if pos> s, we want [s..pos-1], [pos..e], so that pos..e part is the "focus" for lowering from right to left?
        # Actually, for lowering to the left, we want to unify intervals from i leftwards. So we might want
        # the [pos.. e] as the "anchor"? But let's keep it consistent with the code approach.

        # now do "lower to the left" from i
        # Because the intervals we unify must include i, which has oldVal or might be partial?
        # But let's ensure that pos is indeed in [s..e].
        lower_left(i, v, pos)

    # We'll now process each query in order
    # For each query k, we do:
    #   1) find i = find_interval(T[k])
    #   2) let oldVal = intervals[i].val
    #   3) v = G[k] - T[k]
    #   4) if v == oldVal => cost=0 => continue
    #      if v > oldVal => do_raise(i, T[k], v)
    #      if v < oldVal => do_lower(i, T[k], v)

    for k in range(Q):
        tpos = T[k]  # 1-based person index
        v = G[k] - tpos  # the new f-value for that person
        i = find_interval(tpos)  # find the interval containing tpos
        s, e, oldVal = intervals[i]
        if oldVal == v:
            # no cost, do nothing
            continue
        elif v > oldVal:
            do_raise(i, tpos, v)
        else:  # v < oldVal
            do_lower(i, tpos, v)

    print(total_cost)