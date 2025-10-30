def solve():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    AB = data[2:]
    
    # Read all functions
    funcs = [(int(AB[2*i]), int(AB[2*i+1])) for i in range(N)]
    
    # Because A_i, B_i ≤ 50, there are at most 2500 distinct (A,B) pairs.
    # Gather counts of each distinct (A,B).
    from collections import Counter
    cnt = Counter(funcs)
    distinct = [(a, b, c) for (a, b), c in cnt.items()]
    # Sort primarily by descending A, secondarily by descending B
    distinct.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # We will use a Branch-and-Bound (DFS) approach with an upper-bound check.
    # Composition: if current f(x) = A_cur*x + B_cur,
    # then choosing next function (A_i, B_i) => new f(x) = (A_cur*A_i)*x + (B_cur + A_cur*B_i).
    #
    # After K steps, the final value at x=1 is A_cur + B_cur.
    #
    # We'll keep a global "best" and prune branches whose possible upper bound
    # cannot exceed the current best.

    # Precompute a quick bounding function: given (A_cur, B_cur, steps_done),
    # greedily pick up to K-steps_done pairs (in descending A order) to estimate
    # the maximum possible final A + B we could achieve if we continue perfectly.
    
    # This bounding does not necessarily reflect the actual order constraints
    # (because to maximize B we want large A earlier), but it's a good over-approximation
    # that is simple and fast to compute. It should still prune many branches.
    
    # Steps for bounding:
    #  1) copy local counts of distinct pairs
    #  2) from the start (A_cur, B_cur), pick up to (K-steps_done) times the pairs
    #     from largest A to smallest, each time updating (A_cur, B_cur).
    #  3) the result A_cur + B_cur is the upper bound.

    def compute_bound(Ac, Bc, used_counts, start_index, steps_done):
        # We have K - steps_done moves left
        r = K - steps_done
        if r == 0:
            return Ac + Bc
        
        tempA, tempB = Ac, Bc
        for i in range(start_index, len(distinct)):
            if r <= 0:
                break
            a, b, avail = distinct[i]
            # how many times can we still pick this pair?
            # we can use up to 'avail - used_counts[i]' times, but not more than r
            can_use = avail - used_counts[i]
            if can_use <= 0:
                continue
            take = min(can_use, r)
            # We'll "simulate" taking it 'take' times in a row
            # (this is an overestimate, but simpler).
            # Actually picking it 'take' times in a row:
            #   After 1 pick: (tempA*a, tempB + tempA*b)
            #   After 2 picks: ...
            # but let's do it one-by-one to keep it correct.
            while take > 0 and r > 0:
                r -= 1
                used_counts[i] += 1
                # compose
                tempB = tempB + tempA * b
                tempA = tempA * a
                take -= 1
        # revert the usage
        return_val = tempA + tempB
        
        # Roll back changes in used_counts
        # (we must undo exactly how many we used).
        # We can do it carefully by re-walking:
        steps_back = K - steps_done - r  # number of times we updated
        idx = start_index
        while steps_back > 0 and idx < len(distinct):
            a, b, avail = distinct[idx]
            can_remove = min(used_counts[idx] - 0, steps_back)
            while can_remove > 0:
                used_counts[idx] -= 1
                steps_back -= 1
                can_remove -= 1
            idx += 1
        
        return return_val

    best_val = 0
    
    # We'll implement a depth-first search with backtracking.
    # We'll keep track of usage of each distinct pair so as not to exceed counts.
    # We'll explore children in descending order of immediate A (and B),
    # and prune when an upper bound can't exceed our current best.

    # To speed up repeated bounding calls, we can keep the used_counts array
    # and pass an index from which we are allowed to pick next.
    # Because picking from earlier index again might still be allowed if there's
    # more availability. So we won't force strictly increasing index usage;
    # we can pick from the entire range each time (since any order is allowed).
    # But we will pass start_index = 0 always to allow picking from anywhere.

    def dfs(Ac, Bc, steps_done, start_index, used_counts):
        nonlocal best_val
        
        if steps_done == K:
            # Update best
            val = Ac + Bc
            if val > best_val:
                best_val = val
            return
        
        # Quick bounding
        bound = compute_bound(Ac, Bc, used_counts, 0, steps_done)
        if bound <= best_val:
            return
        
        # Try picking each available pair
        # (sorted in descending (A,B)), to keep a "larger-first" exploration
        for i in range(len(distinct)):
            a, b, avail = distinct[i]
            if used_counts[i] < avail:
                # pick
                used_counts[i] += 1
                newB = Bc + Ac * b
                newA = Ac * a
                dfs(newA, newB, steps_done + 1, 0, used_counts)
                used_counts[i] -= 1

    # We start with identity: f(x) = x => (A=1, B=0)
    used_counts = [0]*len(distinct)
    dfs(1, 0, 0, 0, used_counts)
    
    print(best_val)

def solve_wrapper():
    solve()

# Optional: call solve() if you want to run it directly here.
# solve()