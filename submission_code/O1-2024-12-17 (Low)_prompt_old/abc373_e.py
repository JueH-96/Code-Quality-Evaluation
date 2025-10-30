def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[:3])
    A = list(map(int, input_data[3:]))

    # -----------------------------------------------------------------------
    # Problem restatement (in our own words):
    #
    # We have N candidates, each currently has A_i votes. The total number of
    # votes cast is K, so there are R = K - sum(A) votes still unallocated.
    #
    # After all votes are allocated, a candidate i "wins" (is elected) if
    # strictly fewer than M other candidates have a strictly larger vote tally
    # than candidate i. Equivalently, the number of candidates whose final
    # vote count is greater than candidate i's final count is < M.
    #
    # For each candidate i, we want the smallest number X (0 <= X <= R) of
    # the remaining votes given to i (and the rest distributed in the worst
    # possible way among the other candidates) so that i is guaranteed to
    # be among the winners no matter how the other R - X votes are distributed.
    #
    # If X=0 already guarantees i's win, output 0. If no nonnegative X<=R
    # can guarantee i's win, output -1.
    #
    # -----------------------------------------------------------------------
    #
    # Key idea for checking if "X extra votes to i guarantees i's win":
    #
    # Let final_i = A_i + X be i's final votes. We want to ensure that in
    # the worst-case distribution of the remaining (R - X) votes among the
    # other (N-1) candidates, fewer than M candidates can end up with a
    # strictly larger total than final_i.
    #
    # Equivalently, if it is *possible* for at least M other candidates to
    # exceed (A_i + X), then i is *not* guaranteed. If it is *impossible*
    # for M others to exceed i, then i is guaranteed.
    #
    # How do we check if M others can exceed (A_i + X)?
    #   For each other candidate j, let needed_j = max(0, (A_i + X + 1) - A_j).
    #   This is the number of votes j needs from the leftover pool to get
    #   strictly more votes than i. If needed_j=0, j already strictly exceeds i.
    #
    # We ask: can we pick a set of size >= M among the other candidates so that
    # each candidate in that set receives at least needed_j votes and the total
    # assigned to them is <= (R - X)?
    #
    # The worst-case for i is that the remaining (R - X) votes are allocated
    # precisely to maximize the number of candidates surpassing i. So we see
    # how many can surpass i if we allocate leftover votes in the best possible
    # way from their perspective.
    #
    # Implementation approach:
    #   1) Sort all A_j (j != i) or use a data structure to handle removal of A_i.
    #   2) Let final_i = A_i + X. Count how many j have A_j > final_i; those j
    #      already strictly exceed i immediately (call that count Q0).
    #      - If Q0 >= M, then i is certainly not guaranteed (fail).
    #   3) Among the rest (those with A_j <= final_i), define needed_j = (final_i+1) - A_j.
    #      Sort these needed_j in ascending order, then greedily see how many
    #      we can fund with the leftover (R - X). If we can fund at least
    #      (M - Q0) of them, that means total surpassers >= M => not guaranteed.
    #      If we cannot fund (M - Q0), that means total surpassers < M => guaranteed.
    #
    # We want the minimal X for which we get "guaranteed." If none, answer -1.
    #
    # Doing a per-candidate naive approach is O(N^2 log N), which is too large.
    #
    # Efficient approach:
    #   We'll do a (balanced) "removal" approach. For each candidate i, we remove A_i
    #   from the data structure, then do a binary search on X in [0..R].
    #   Each check is O(log N) using prefix sums/fenwicks for:
    #     - Q0 = # of A_j >= (A_i + X + 1)
    #     - Sum of the smallest (M - Q0) needed_j among A_j <= (A_i + X).
    #
    # The binary search takes up to ~log2(R) ~ 40 steps. Times N=2e5 => 8e6 checks,
    # each check ~ O(log N) => ~ 1.6e8. This is borderline in Python.
    #
    # However, we can prune:
    #   - If a candidate i already has so many votes that # of j with A_j > A_i is < M
    #     and cannot reach M even if those j get all leftover R, i needs 0. We do that check quickly.
    #   - Conversely, if # of j with A_j + R > A_i is already >= M, i can never surpass them => -1.
    #
    # After these quick checks, only borderline candidates remain. We then do the binary search.
    #
    # Let's implement with these optimizations and a Fenwick tree (or prefix sums + frequency).
    #
    # Implementation outline steps:
    #
    #   1) Let S = sum(A). R = K - S. Pre-check i:
    #       - Count how many j have A_j > A_i + R. If that count >= M => C_i = -1 (cannot fix).
    #       - Count how many j have A_j + R < A_i. If that count <= (M-1), that means at most that many can exceed i, so i is already safe => C_i = 0.
    #       Else we proceed to the binary search approach.
    #
    #   2) We set up a frequency array of A_j (j != i) to query Q0 and partial sums for needed_j.
    #      We'll do it by having a global freq array for all A values, then freq[A_i]-- ( removing i ), do the check, and freq[A_i]++.
    #
    #   3) The binary search for X in [0..R].
    #      - mid = (low + high)//2
    #      - final_i = A_i + mid
    #      - Q0 = number of j with A_j > final_i  => if Q0 >= M => we fail => low=mid+1
    #        else we define need_count = M - Q0. If need_count <= 0 => pass => high=mid
    #        else let sum_needed = sum of the need_count smallest values of (final_i+1 - A_j)
    #             among those A_j <= final_i; if sum_needed <= (R - mid) => fail => low=mid+1
    #             else pass => high=mid
    #      - answer = low if low<=R else -1
    #
    # Implementation details: We'll do coordinate compression for all possible A's,
    # build a Fenwick or prefix-sum structure. Then for each i, we do freq[A_i]--,
    # do the pre-check, if borderline then do the binary search with queries
    # using the Fenwick/prefix sums. Then freq[A_i]++.
    #
    # Let's implement carefully. This is advanced and tight on time, so we will
    # implement as efficiently as possible in Python. In practice, C++ would be safer.
    #
    # -----------------------------------------------------------------------

    # Quick pre-check aggregates:
    # Sort the A array. We'll use it for rank queries.
    sortedA = sorted(A)
    from bisect import bisect_left, bisect_right

    S = sum(A)
    R = K - S  # leftover votes

    # Fenwick (Binary Indexed Tree) for frequencies and Fenwick for prefix sums:
    #  - We will coordinate-compress the values of A.
    #  - We'll store freq in fenwicks for counts and for sum of actual values.

    # Coordinate compression
    unique_vals = []
    # We'll gather all A in a list
    unique_vals = sorted(set(A))
    # We'll map each A_i to a rank in [1..len(unique_vals)] for Fenwicks to be 1-indexed.
    rank_map = {}
    for idx, val in enumerate(unique_vals):
        rank_map[val] = idx+1

    max_rank = len(unique_vals)

    # Fenwicks
    # fenw_count[r] = sum of frequencies of ranks <= r
    # fenw_sum[r]   = sum of values (original) for ranks <= r
    def fenw_update(fenw, pos, delta):
        while pos <= max_rank:
            fenw[pos] += delta
            pos += pos & -pos

    def fenw_sum_(fenw, pos):
        s = 0
        while pos>0:
            s += fenw[pos]
            pos -= pos & -pos
        return s

    def fenw_range_sum(fenw, l, r):
        if r<l: return 0
        return fenw_sum_(fenw, r) - fenw_sum_(fenw, l-1)

    fenw_count = [0]*(max_rank+1)
    fenw_vals  = [0]*(max_rank+1)

    # Build initial fenwicks from all frequencies
    freq = [0]*(max_rank+1)
    for v in A:
        r = rank_map[v]
        freq[r]+=1
    for r in range(1, max_rank+1):
        fenw_update(fenw_count, r, freq[r])
        fenw_update(fenw_vals,  r, freq[r]*unique_vals[r-1])

    # Helper to remove/add a single value v from fenwicks
    def remove_value(v):
        rr = rank_map[v]
        fenw_update(fenw_count, rr, -1)
        fenw_update(fenw_vals,  rr, -v)

    def add_value(v):
        rr = rank_map[v]
        fenw_update(fenw_count, rr, 1)
        fenw_update(fenw_vals,  rr, v)

    # Function to get:
    #   count_ge(x) = number of elements >= x
    # We can get count_le(x-1) and subtract from total. We do rank on (x-1).
    def count_ge(x):
        # we want #A_j >= x
        # find r0 = the smallest rank s.t. unique_vals[r0-1] >= x
        # if all are < x => return 0
        # else return total_count - fenw_count(r0-1)
        # We'll do bisect_left for x in unique_vals
        r0 = bisect_left(unique_vals, x) + 1  # +1 for fenw indexing
        total = fenw_sum_(fenw_count, max_rank)
        if r0 > max_rank:
            # all are < x
            return 0
        else:
            return total - fenw_sum_(fenw_count, r0-1)

    # Similarly, we can find sum of the top k elements among those <= T for some T,
    # by taking count_le(T) = c, then we want the sum of the largest k among those c elements.
    #
    # If c < k, we can't pick k distinct elements from those <= T, so the sum of largest k is not fully definable.
    # But we are using it to check how many can surpass i if we give them leftover votes, so we'll handle that logic carefully.

    # We'll implement a helper: sum_of_largest_k_le(T, k).
    # This uses that sum of all <= T is fenw_val up to rank_of_T
    # We'll do a binary search approach to gather from the top down. Since we only need it once per check,
    # let's implement a function that does a partial "binary search" in the Fenwick structure to find
    # which ranks we are picking from and how many freq from each rank, going from the highest rank
    # that is <= T downwards.

    def count_le(T):
        # number of elements <= T
        # find rT = bisect_right(unique_vals, T)
        # then fenw_sum(fenw_count, rT)
        rT = bisect_right(unique_vals, T)
        return fenw_sum_(fenw_count, rT)

    def sum_le(T):
        # sum of elements <= T
        rT = bisect_right(unique_vals, T)
        return fenw_sum_(fenw_vals, rT)

    # sum_of_largest_k_le(T,k):
    #   We'll pick from the max rank <= T downwards until we gather k items or exhaust.
    #   We'll do a "backwards Fenwick" approach.  This is not a standard routine, so we'll implement a small loop:
    def sum_of_largest_k_le(T, k):
        # if there are c = count_le(T) items <= T, and c <= k => we sum all of them
        c = count_le(T)
        if c <= k:
            return sum_le(T)
        # else we only want partial. We'll do repeated binary search to find the highest rank with frequency, use as many as possible, etc.
        # We'll do a manual loop using Fenwicks. This must be done carefully but quickly.

        # We'll find the rank rT of T for easier partial sum:
        rT = bisect_right(unique_vals, T)
        remain = k
        result = 0
        # We'll descend from rT down to 1 in Fenwicks, picking freq at each rank.
        # A known Fenwick trick to iterate ranks in descending order:
        #   but we'll do it using a while remain>0:
        curr = rT
        while remain>0 and curr>0:
            # We'll find the largest step we can do with Fenwicks. But simpler is to
            # "drill down" - we find the position in Fenwicks that holds the partial sum >= ...
            # Actually we want to pick from the rank = curr if freq>0.
            # Then reduce remain by min(freq[curr], remain).
            # Then move curr--
            # That would be O(rT) in worst case. Which might be big (up to 2e5).
            # But we only do this after the binary search step concluded c>k, so k < c. In the worst case, this is still O(N).
            #
            # We'll do a slightly more advanced approach: We'll jump by Fenwicks if we can, but implementing
            # a standard "find by Fenwicks" is usually in ascending mode. We can do it in ascending mode to find
            # how many we can skip from the bottom. This is complicated.
            #
            # For simplicity in Python (time is short), we can do a standard approach:
            #   We'll do a repeated binary search to find the largest rank rX <= curr for which fenw_count cumulative is >= something.
            #   This might lead to a log(N) factor each time we pick 1 item. That could be up to k times => k log N, which is worst 2e5 log(2e5) => 2.2e6, might be borderline but might pass with fast IO.
            #
            # Alternatively, we can do a partial gather approach in O(log N) to pick remain items from the top, but that is more complex to implement reliably. Let's do the simpler approach with caution.
            #
            # We'll do a more direct approach:
            #   1) Find how many items are in rank=curr: fcurr = freq rank=curr => fenw_range_sum(fenw_count, curr, curr)
            #   2) If fcurr>0:
            #        pick used = min(remain, fcurr)
            #        result += used * unique_vals[curr-1]
            #        remain -= used
            #        update Fenwicks to reduce freq at curr by used
            #      curr -= 1
            #
            # Then after finishing, restore frequencies. This is destructive. We must revert at the end.
            #
            # This is the simplest correct approach to implement here. We'll do it carefully and restore after.
            fcurr = fenw_range_sum(fenw_count, curr, curr)
            if fcurr>0:
                used = min(remain, fcurr)
                result += used * unique_vals[curr-1]
                remain -= used
                # update Fenwicks
                fenw_update(fenw_count, curr, -used)
                fenw_update(fenw_vals,  curr, -(used*unique_vals[curr-1]))
            else:
                curr -= 1

        # now we used up k items
        # restore frequencies by adding them back
        # We'll do a second pass, but we need how many we removed from each rank. Let's store them.
        # We'll do the partial process again but this time storing how many we remove at each step.
        # Actually let's do it in one pass, storing the used amounts in a stack, then revert.

        # We'll implement it properly now (with storing usage) to revert after the sum is computed.
        return result

    # We'll do a single function that gets the sum of the largest k among those <= T, but doesn't destroy the data structure.
    # We'll code a subroutine that does the destructive approach, returns result, and reverts.

    def sum_of_largest_k_le_safe(T, k):
        c = count_le(T)
        if c <= k:
            return sum_le(T)
        # partial pick
        rT = bisect_right(unique_vals, T)
        remain = k
        result = 0
        usage_stack = []

        curr = rT
        while remain>0 and curr>0:
            fcurr = fenw_range_sum(fenw_count, curr, curr)
            if fcurr>0:
                used = min(remain, fcurr)
                result += used * unique_vals[curr-1]
                remain -= used
                if used>0:
                    usage_stack.append((curr, used))
                # apply removal
                fenw_update(fenw_count, curr, -used)
                fenw_update(fenw_vals,  curr, -(used*unique_vals[curr-1]))
            else:
                curr -= 1

        # revert
        for (rr, uu) in usage_stack:
            fenw_update(fenw_count, rr, uu)
            fenw_update(fenw_vals,  rr, uu*unique_vals[rr-1])

        return result

    total_count = fenw_sum_(fenw_count, max_rank)  # = N

    # Precompute quick checks for each candidate i:
    #  - Let out_count = # of j with A_j > A_i + R
    #    If out_count >= M => C_i = -1
    #  - Let in_count = # of j with A_j + R < A_i
    #    If in_count <= (M-1) => i is already safe => C_i = 0
    #
    # Otherwise do a binary search.
    results = [None]*N

    # We'll define a function that checks if X is enough for i:
    #   1) final_i = A_i + X
    #   2) Q0 = count of j with A_j > final_i
    #      if Q0 >= M => return False (fail)
    #   3) need_count = M - Q0
    #      if need_count <= 0 => return True (pass)
    #   4) sum_need = sum_of_smallest need_count of [ (final_i+1)-A_j for j with A_j<=final_i ]
    #      but that sum_of_smallest is actually sum_of_largest need_count among A_j in the range [0..final_i]
    #      in terms of A_j. Because cost = (final_i+1 - A_j). The largest A_j => smallest cost.
    #      sum_need = need_count*(final_i+1) - (sum of largest need_count A_j <= final_i).
    #      We'll compute sum_of_largest_k_le_safe(final_i, need_count).
    #      Then cost = need_count*(final_i+1) - that_sum.
    #      if cost <= R - X => we can fund them => fail, else pass
    #
    # We'll write a small sub-check function do_check(X, Ai).
    # We'll do it after removal of Ai from fenwicks.

    def can_guarantee(i, X):
        fi = A[i] + X
        # Q0
        Q0 = count_ge(fi+1)
        if Q0 >= M:
            return False
        need_count = M - Q0
        if need_count <= 0:
            return True
        # sum_of largest need_count
        bigsum = sum_of_largest_k_le_safe(fi, need_count)
        # cost = need_count*(fi+1) - bigsum
        cost = need_count*(fi+1) - bigsum
        if cost <= R - X:
            return False
        return True

    # We'll implement "solve_candidate(i)" that returns the needed X or 0 or -1.
    # Steps:
    #   1) remove A[i] from fenwicks
    #   2) do quick checks
    #   3) if borderline => do binary search
    #   4) add A[i] back
    #
    # We'll implement it:

    def solve_candidate(i):
        # remove A[i]
        remove_value(A[i])

        ai = A[i]
        # out_count = # j with A_j > A[i] + R
        out_count = count_ge(ai + R + 1)
        if out_count >= M:
            # cannot fix, answer -1
            add_value(ai)
            return -1

        # in_count = # j with A_j + R < A[i], i.e. # j with A_j < A[i] - R, but that doesn't make sense.
        # Actually we want # j that cannot surpass i even if they get all R votes:
        # condition for j to surpass i is: A_j + R >= A[i] => then j can tie or surpass i with the right distribution
        # so if A_j + R < A[i], j definitely cannot surpass i. Let in_count = # j with A_j + R < A[i].
        # if the number of j that definitely cannot surpass i is N-1, i is trivially in top M? That depends on M though.
        #
        # Actually let's count how many can definitely surpass i if i got 0 additional votes. Not the same.
        #
        # The official pre-check from the statement of approach:
        #   if # of j with A_j + R < A_i => those j can never catch i, so they are not a threat.
        #   Let threatened = number of j that can possibly surpass i => # j with A_j + R >= A_i
        #   If threatened <= M-1, that means at most threatened can surpass i, so i is guaranteed top M => 0.
        threatened = 0
        # number of j with A_j + R >= A_i => A_j >= A_i - R
        # so threatened = # of j with A_j >= A_i - R
        c_le = 0
        if ai - R > -10**15:  # just guarding bisect
            c_le = bisect_left(unique_vals, ai - R)  # # of unique_vals < (ai-R)
        # Actually we want # j with A_j >= (ai-R).
        # count of j with A_j < (ai-R) is c_le, so threatened = total_count-1 - c_le? We need an exact fenw query:
        # simpler: threatened = # j with A_j >= (ai-R).
        # do a function count_ge(ai-R).
        # But we have already removed A[i], so the total is now N-1 in fenwicks.
        # We'll do threatened = N-1 - # j with A_j < (ai-R).
        # # j with A_j < (ai-R) = fenw_sum for ranks < rank_of(ai-R).
        # We'll build a small function to do count_ge(t) with the fenwicks. We have it. So:
        threatened = count_ge(ai - R)
        # If threatened <= M-1 => i is guaranteed with 0 extra votes => return 0
        if threatened <= M-1:
            add_value(ai)
            return 0

        # Otherwise do a binary search over X in [0..R].
        low, high = 0, R+1
        while low<high:
            mid = (low+high)//2
            if can_guarantee(i, mid):
                high = mid
            else:
                low = mid+1
        ans = low
        if ans>R:
            ans = -1

        # revert
        add_value(ai)
        return ans

    # Compute answers
    # We will loop i in [0..N-1], compute result
    # Print them
    out = []
    for i in range(N):
        out.append(str(solve_candidate(i)))

    print(" ".join(out))