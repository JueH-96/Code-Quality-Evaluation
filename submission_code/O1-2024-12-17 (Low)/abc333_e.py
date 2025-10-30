def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    tx = input_data[1:]  # pairs t_i, x_i

    # Separate out the events into (t_i, x_i) in 0-based indexing
    events = [(int(tx[2*i]), int(tx[2*i+1])) for i in range(N)]

    # ------------------------------------------------------------
    # 1) Quick feasibility check: for each monster of type x, there
    #    must exist at least one potion of type x before it in the
    #    event sequence. If there's a monster with no such potion,
    #    answer is -1 immediately.
    #
    #    More precisely: let Mx_count[x] = number of monsters of type x
    #    let Px_positions[x] = sorted list of positions where a potion
    #      of type x appears
    #    as we encounter the k-th monster of type x in chronological order,
    #    we need at least k potions among positions < monster_i. If that
    #    is not satisfied, it's impossible.
    # ------------------------------------------------------------

    from collections import defaultdict

    # Count total monsters of each type and gather potion positions
    monster_positions_by_type = defaultdict(list)
    potion_positions_by_type = defaultdict(list)
    monster_count_by_type = defaultdict(int)
    for i, (t, x) in enumerate(events):
        if t == 1:
            potion_positions_by_type[x].append(i)
        else:
            monster_positions_by_type[x].append(i)

    # For feasibility, check that for each type x, for the j-th monster
    # we have at least j potions in positions < that monster.
    # Because if the j-th monster occurs at index mpos, we need at least j potions
    # from potion_positions_by_type[x] that are < mpos.
    # We'll do a simple merge of monster positions and potion positions and
    # see if at each monster we have enough potions strictly earlier.

    for x, mpositions in monster_positions_by_type.items():
        pot_positions = potion_positions_by_type[x]
        pot_positions.sort()
        mpositions.sort()
        # We just need for monster j (0-based) at index mpos,
        # there must be at least j+1 potions [strictly] less than mpos.
        # i.e. pot_positions[j] < mpos. If j >= len(pot_positions), fail
        # or if pot_positions[j] >= mpos, fail.
        for j, mpos in enumerate(mpositions):
            if j >= len(pot_positions):
                print(-1)
                return
            if pot_positions[j] >= mpos:
                print(-1)
                return

    # If we pass the above check, it is possible to defeat all monsters
    # in some way. Now we need to find the minimal K (maximum inventory)
    # among all valid strategies.
    #
    # ------------------------------------------------------------
    # 2) We will use a binary-search over K from 0..(some upper bound).
    #    A valid upper bound is N (picking all potions). Then we check
    #    feasibility of each K by a left-to-right "greedy" picking:
    #
    #    We track:
    #      needed[x] = how many *future* monsters of type x remain
    #      holding[x] = how many potions of type x we currently hold
    #      current_inventory = sum of all holding[x] over x
    #
    #    We process events in order:
    #      - If it's a monster of type x, we must have holding[x] > 0 or fail.
    #        Then holding[x]--, current_inventory--, needed[x]--.
    #      - If it's a potion of type x, we decide whether to pick it or not.
    #        We only need to pick it if we still haven't guaranteed enough potions
    #        for the future monsters of type x. In other words, if holding[x]
    #        + (the number of potions of type x that will appear later) < needed[x],
    #        then we must pick this potion. But picking it cannot cause
    #        current_inventory to exceed K. If it would, we fail for this K.
    #        Otherwise we can skip it if we already have enough potions for
    #        future x-monsters (or if picking would exceed K).
    #
    #    If we get through all events without failing, then we can do with K.
    #    Then we lower K; else we increase K. We find the minimal K that works.
    #
    #    Finally, we do one more pass with that K to record exactly which potions
    #    we pick (for the output).
    #
    #    This standard approach works because:
    #     - We only pick a potion if it is *forced* by the future monster demands.
    #     - By never picking unneeded potions (and never skipping needed ones),
    #       we ensure the inventory is as small as possible for that K.
    #
    # ------------------------------------------------------------

    # Precompute how many monsters of each type remain AFTER each index i.
    # This helps us quickly decide if we "must" pick a potion at event i.
    from collections import defaultdict

    # Count total monsters of each type
    total_monsters = defaultdict(int)
    for t, x in events:
        if t == 2:
            total_monsters[x] += 1

    # Monsters remaining after index i. We'll build suffix arrays:
    # remain[x][i] = number of monsters of type x in events i..N-1
    # we can do this by a single pass from right to left
    remain = defaultdict(lambda: [0]*(N+1))
    # remain[x][N] = 0 for all x
    # fill from the right
    # for i in reversed range(N): remain[x][i] = remain[x][i+1] + (1 if events[i] is monster of type x else 0)
    for x in total_monsters:
        remain[x] = [0]*(N+1)

    # Build them
    # We'll go from right to left, and keep a running count for each x
    running_count = defaultdict(int)
    # i from N-1 down to 0
    for i in range(N-1, -1, -1):
        t, x = events[i]
        for key in total_monsters:
            remain[key][i] = remain[key][i+1]
        if t == 2:
            remain[x][i] += 1

    # Now we define a function feasible(K) that tries to do the pass
    # and returns True/False depending on if we can avoid holding
    # more than K potions at once, and also defeat all monsters.
    def can_do_with_K(K):
        # holding[x] = how many potions of type x we currently have
        holding = defaultdict(int)
        current_inv = 0
        for i, (t, x) in enumerate(events):
            if t == 2:
                # must have a potion
                if holding[x] == 0:
                    return False
                # use one
                holding[x] -= 1
                current_inv -= 1
            else:
                # t == 1, a potion found
                # check if we still need it
                # needed_for_x = remain[x][i+1], how many x-monsters remain after i
                # if holding[x] + possible_future_potions < needed_for_x, must pick
                # possible_future_potions = number of potions of type x in events i+1..N-1
                # but we don't have that precomputed directly. We could do so if needed.
                # Or simpler: we know total_monsters[x] how many x-monsters total,
                # and we can track how many of them we've already faced. But that's
                # not enough to tell how many potions of type x will appear later.
                #
                # We'll do another pass to count how many potions of type x remain
                # after i. Precompute similarly to remain.

                # We want to see how many x-monsters remain after i: remain[x][i+1].
                # We also want how many potions of type x remain after i, call it pot_remain[x][i+1].
                # If holding[x] + pot_remain[x][i+1] < remain[x][i+1], we MUST pick now.
                # If we pick, we must not exceed K. If current_inv == K, then fail.
                #
                # Because of time constraints, let's build pot_remain as well.
                pass
        return True

    # However, we also need a parallel array pot_remain[x][i] for each type x.
    # Let's build that similarly to remain, in a dictionary.

    # First, count how many potions of type x total
    total_potions = defaultdict(int)
    for t, x in events:
        if t == 1:
            total_potions[x] += 1
    # pot_remain[x] = array of length N+1
    pot_remain = defaultdict(lambda: [0]*(N+1))
    # build them from right to left
    for x in total_potions:
        pot_remain[x] = [0]*(N+1)

    running_pot = defaultdict(int)
    for i in range(N-1, -1, -1):
        t, x = events[i]
        for key in total_potions:
            pot_remain[key][i] = pot_remain[key][i+1]
        if t == 1:
            pot_remain[x][i] += 1

    def feasible(K):
        holding = defaultdict(int)
        current_inv = 0
        # We'll also store how many of each type of monster we've used potions on so far
        # but for decisions we only need remain[x][i+1] vs pot_remain[x][i+1].
        for i, (t, x) in enumerate(events):
            if t == 2:
                # Monster
                if holding[x] == 0:
                    return False
                holding[x] -= 1
                current_inv -= 1
            else:
                # Potion
                # Must pick if holding[x] + pot_remain[x][i+1] < remain[x][i+1]
                # i.e. if we do NOT pick this potion, the maximum number of potions
                # of type x we can have ( now + future ) is holding[x] + pot_remain[x][i+1 ],
                # which might be < how many monsters remain[x][i+1]. Then we'd fail eventually,
                # so we must pick this potion now.
                must_pick = False
                if holding[x] + pot_remain[x][i+1] < remain[x][i+1]:
                    must_pick = True
                if must_pick:
                    if current_inv == K:  # can't pick, would exceed K
                        return False
                    holding[x] += 1
                    current_inv += 1
                else:
                    # we skip
                    pass
        return True

    # We'll do a binary search from 0..N
    left, right = 0, N+1
    okK = None
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            okK = mid
            right = mid
        else:
            left = mid+1

    if okK is None or okK == N+1:
        # Should not happen if feasibility check above passed, but just in case:
        print(-1)
        return

    # We have found K_min = okK
    K_min = okK

    # ------------------------------------------------------------
    # 3) Reconstruct which potions we pick with K_min
    #    We'll do the same pass, but record the decisions now.
    # ------------------------------------------------------------

    pick_decision = [0]*N  # 1 if we pick the potion at event i, 0 otherwise

    holding = defaultdict(int)
    current_inv = 0
    for i, (t, x) in enumerate(events):
        if t == 2:
            # monster, use one
            holding[x] -= 1
            current_inv -= 1
        else:
            # potion
            if holding[x] + pot_remain[x][i+1] < remain[x][i+1]:
                # must pick
                pick_decision[i] = 1
                holding[x] += 1
                current_inv += 1
            else:
                # skip
                pass

    # Double-check we never exceeded K_min in the reconstruction
    # (it should not if feasible() is correct)
    # We'll do a quick forward pass to be sure:
    # But it should be guaranteed by the logic. We'll trust feasibility().

    # ------------------------------------------------------------
    # 4) Output the result
    #    First line: K_min
    #    Second line: pick decisions for each event i with t_i=1, in ascending order of i.
    # ------------------------------------------------------------

    print(K_min)
    ans = []
    for i, (t, x) in enumerate(events):
        if t == 1:
            ans.append(str(pick_decision[i]))
    print(" ".join(ans))