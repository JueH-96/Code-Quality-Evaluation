def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    S = input_data[2]
    T = input_data[3]
    
    # Quick edge case: if M=1, then T has exactly one character.
    # We can form S iff S is all the same character as T[0].
    if M == 1:
        if all(ch == T[0] for ch in S):
            print("Yes")
        else:
            print("No")
        return
    
    #--------------------------------------------------------------------------
    # GENERAL CASE (M up to 5)
    #
    # We are allowed to overwrite any M-consecutive block of '#' with T any
    # number of times, in any order.  We want to know if we can end up with S.
    #
    # A well-known way (and one that lines up with editorial hints for problems
    # like this) is to think of "which offset of T sets each position of S in
    # the final over-write order?"  Because M is small, we can do a constraint-
    # satisfaction check with "local" constraints among positions that are
    # within distance M-1 of each other.
    #
    # WHY DOES THIS WORK?
    #  - In the final arrangement, each position i is ultimately written by
    #    exactly one "topmost" T-block.  That means: if that block starts at j,
    #    then i-j = offset_in_T, and T[offset_in_T] == S[i].
    #  - However, any other block that covers i and comes *after* this block
    #    in the overwrite order must also write the same S[i], otherwise
    #    S[i] would be spoiled.  In effect, either we do not cover i again,
    #    or if we do, we must use the same offset letter for i.
    #  - The only real conflicts happen when two positions i < p < i+M overlap
    #    each other in blocks, forcing one block to come after the other.  One
    #    can show that all global conflicts boil down to local pairwise
    #    constraints among positions within distance < M.  Because M ≤ 5,
    #    each position only has up to 4 neighbors to check.
    #
    # ALGORITHM:
    #  1) Build "offsetSet[ch]" = all offsets k in [0..M-1] s.t. T[k] == ch.
    #     If some S[i] has no offset in T, it's impossible => "No."
    #
    #  2) For each i, we keep a small set (bitmask) of possible offsets
    #     that can ultimately write S[i].  Initially that is offsetSet[S[i]].
    #
    #  3) We impose "no-conflict" constraints among positions i < p < i+M.
    #     Let gap = p - i.  Offsets (o_i, o_p) are *compatible* if there exists
    #     some ordering of the two final blocks that does not spoil S[i] or S[p].
    #     Concretely:
    #        - If a block sets i with offset o_i, it covers positions
    #          [i - o_i .. i - o_i + M - 1].
    #          That block will also overwrite p if p is in its range.  That
    #          happens exactly if (p - i) < M and p >= i => indeed gap < M.
    #          Within that block, the offset for p would be (p - (i - o_i)) = o_i + (p - i).
    #          But we only *care* if that block is topmost over p.  If that block
    #          is *not* topmost, then a later block can fix p.  So there's only
    #          a conflict if the block that sets i is placed *after* the block
    #          that sets p, yet it also inadvertently overwrites p incorrectly.
    #          Similarly for the other block (the one that sets p).
    #
    #       A simpler, known sufficient check is:
    #         Let gap = p - i (1 <= gap <= M-1).
    #         For offsets (o_i, o_p) to have no irreconcilable conflict, they
    #         must satisfy "there is some ordering of the two blocks that
    #         does not break S[i] or S[p]."  This equates to:
    #           EITHER
    #             the block for i does NOT cover p => o_i < gap
    #             (so i's block is too far left to reach p),
    #             OR
    #             the block for p does NOT cover i => o_p < gap
    #             (so p's block is too far left to reach i if p>i),
    #             OR
    #             they both do cover each other => o_i >= gap and o_p >= gap
    #             AND in that overlap we must have T[o_i - gap] == S[p] and
    #                                     T[o_p - gap] == S[i].
    #
    #  4) We then run a standard "arc-consistency" or "iterative pruning" approach
    #     over these local constraints.  Because each position i has at most M
    #     possible offsets, we store them in a bitmask.  Then for each pair
    #     (i, p) with p-i < M, we store which pairs (o_i, o_p) are valid.  In
    #     code, we actually store from i->p: for each o_i, a bitmask of possible
    #     o_p that remain valid.  Then we do iterative pruning: if we remove
    #     o_i from i, we see how that affects p, and so on.  If ever a position
    #     runs out of valid offsets, answer "No."  If we stabilize, answer "Yes."
    #
    # This is a known technique for small M and large N.
    #
    # Complexity: We have up to N*(M-1) "neighbor edges," each with up to M×M=25
    # potential offset pairs.  We store them in a compressed bitmask form and do
    # a queue-based AC-3 style pruning.  This runs in roughly O(N*M^2) which is
    # acceptable for N up to 2e5 and M=5 (a few million operations in the worst
    # case).
    #
    # Let us implement.
    #--------------------------------------------------------------------------
    
    # Build offsetSet for letters in T
    # offsetSet[L] = all positions k in [0..M-1] s.t. T[k] == L
    # We'll store it as a list or direct bitmask.  But letters are 'A'..'Z',
    # we can do a dict or array of length 26.
    
    from string import ascii_uppercase
    
    # Map letter -> index (0..25)
    def letter_index(c):
        return ord(c) - ord('A')
    
    # For each letter in 'A'..'Z', gather offsets in T
    offset_sets = [[] for _ in range(26)]
    for i, ch in enumerate(T):
        offset_sets[letter_index(ch)].append(i)
    
    # Now build the initial validOffsetsMask for each position i in S
    # validOffsetsMask[i] is a bitmask of length M (<=5)
    # where bit k means "offset k is still possible at position i."
    validOffsetsMask = [0]*N
    
    for i in range(N):
        cidx = letter_index(S[i])
        # All offsets in offset_sets[cidx] are possible *if any*
        # but if there are none, it's impossible
        if not offset_sets[cidx]:
            print("No")
            return
        mask = 0
        for off in offset_sets[cidx]:
            mask |= (1 << off)
        validOffsetsMask[i] = mask
    
    # Precompute a helper: noConflict(o_i, o_p, iLetter=S[i], pLetter=S[p], gap=p-i)
    # We'll turn that into a function that returns True/False.
    def no_conflict(o_i, o_p, s_i, s_p, gap):
        # either block for i doesn't cover p => o_i < gap
        # or block for p doesn't cover i => o_p < gap
        # or both cover => o_i >= gap and o_p >= gap
        #    and T[o_i - gap] == s_p  and T[o_p - gap] == s_i
        if o_i < gap or o_p < gap:
            return True
        # else both >= gap
        if (T[o_i - gap] == s_p) and (T[o_p - gap] == s_i):
            return True
        return False
    
    # Build adjacency:
    # For each i in [0..N-1], for each d in [1.. M-1], if i+d < N,
    # we build a bitmask adj[i][o][d] that says which offsets o' are valid for i+d
    # given offset o for i.
    #
    # We'll store adjacency in a 3D array: adj[i][o][d] = bitmask
    # with up to N*(M)*(M-1).  That is ~2e5 * 5 * 4 = 4e6 entries max,
    # which in Python is large but (just) doable in memory if we are careful.
    # We'll attempt to be as memory-conscious as possible.
    #
    # Then we will do an "arc consistency" iteration.  Whenever we remove offset o
    # from i, we prune offsets from i+d that rely on (o) in adj[i][o][d].
    
    # To avoid huge memory usage of a triple array of size [N][M][M],
    # note that d only goes up to M-1 (which is at most 4). So it's [N][M][4].
    # We'll store that in a list of lists of ints.  We do:
    #    adj[i][d][o] = bitmask of valid offsets in i+d.
    # This is a bit more transposed but simpler to index: adj[i][d][o].
    
    # Initialize
    adj = [ [ [0]*M for _ in range(M) ] for _ in range(N) ]
    # Actually that is adj[i][d][o], but we only need d up to M-1 => we do
    # a 2D array for each i: adj[i][d], where d in [1..M-1].
    # Each adj[i][d] is an array of length M of bitmasks.
    # But storing them as above is too big in Python.  We need to be mindful...
    #
    # Alternatively we can store:
    #   adj[i][d][o] = 0 initially
    # Then fill only if i+d < N.  That is still allocated for all i,d. It's okay.
    
    # We will build them on the fly.  Then use them.  Let's do it carefully.
    
    # For convenience, let's store S as a list of chars to speed up indexing:
    s_list = list(S)
    
    # Build adjacency
    # for i in [0..N-1]:
    #   for d in [1..M-1]:
    #       p = i + d
    #       if p >= N: break
    #       for o in [0..M-1]:
    #           we want to set adj[i][d][o] = bitmask of o' in [0..M-1] s.t.
    #           no_conflict(o, o', s_list[i], s_list[p], d) is True
    # We'll do it once.  Then the arc-consistency will refer to it.
    
    # We'll do it in a top-down manner, mindful of speed in Python.
    T_list = list(T)  # so T_list[x] is T[x]
    
    # Precompute noConflict table for all (o_i, o_p, s_i, s_p, gap) with gap in [1..M-1].
    # But s_i, s_p can be among 26 letters, o_i, o_p in [0..M-1].  That might be big.
    # Instead we'll just check inline.  It's only 25 checks max per adjacency build.
    
    # We'll build for each i a partial adjacency array of length M for d in [1..M-1].
    # Then assign back to adj[i][d].
    
    # Actually, to save memory, let's store adj[i] as a list of length (M-1),
    # each an array of length M of bitmasks.  That is about N*(M-1)*M = 200k*4*5=4e6 ints.
    # We will do:
    #   adj[i][d-1][o]   # index d-1 because d≥1.
    # This will be an integer bitmask of length M.
    adj = [ [ [0]*M for _ in range(M-1) ] for _ in range(N) ]
    
    for i in range(N):
        # for each d in 1..M-1
        max_d = min(M-1, N-1 - i)  # so i+d < N
        for d in range(1, max_d+1):
            p = i + d
            s_i = s_list[i]
            s_p = s_list[p]
            gap = d
            # Build an array of bitmasks "row" of length M.
            # row[o] = sum of 1<<o' for o' in [0..M-1] if no_conflict(o, o', s_i, s_p, gap)
            row = [0]*M
            for o in range(M):
                bitmask = 0
                # only bother if (1<<o) is possible in validOffsetsMask[i]
                # we'll still store it; the AC pass will handle weeding.
                for o_p in range(M):
                    if no_conflict(o, o_p, s_i, s_p, gap):
                        bitmask |= (1 << o_p)
                row[o] = bitmask
            # store row in adj[i][d-1]
            adj[i][d-1] = row
    
    # Now we implement AC-3 style pruning with a queue.
    # We'll keep a queue of positions where an offset was removed.
    # Then propagate constraints to neighbors.
    
    from collections import deque
    Q = deque()
    
    # Initially, just enqueue every position so we can "propagate" once
    for i in range(N):
        Q.append(i)
    
    # We will define a function prune(i) that, given the current validOffsetsMask[i],
    # updates the neighbors i±d for d in [1..M-1] to remove offsets that are no longer
    # supported.
    
    def prune(pos):
        baseMask = validOffsetsMask[pos]
        # For each d in [1..M-1], if pos+d < N, we constrain that for each offset o'
        # in that neighbor, there must be some o in pos's valid set that is "compatible."
        max_d = min(M-1, N-1 - pos)
        for d in range(1, max_d+1):
            nei = pos + d
            oldMask = validOffsetsMask[nei]
            if oldMask == 0:
                continue  # already empty
            # Compute which offsets in 'nei' are still supported by the current offsets in 'pos'.
            # That is the union over all o in baseMask of adj[pos][d-1][o].
            # We'll do a small loop over the bits set in baseMask.
            supported = 0
            bm = baseMask
            while bm:
                # pick lowest set bit
                low = (bm & -bm)
                o = (low).bit_length() - 1  # index of that bit
                bm ^= low
                supported |= adj[pos][d-1][o]
            # Now we intersect with oldMask.  The new valid is oldMask & supported.
            newMask = oldMask & supported
            if newMask != oldMask:
                # Some offsets in neighbor got removed
                removed = oldMask & ~newMask
                validOffsetsMask[nei] = newMask
                if newMask == 0:
                    # No valid offsets -> immediate "No"
                    return False
                # For each bit removed, we queue neighbor
                Q.append(nei)
        
        # Similarly, for d in [1..M-1], if pos-d >= 0
        max_d = min(M-1, pos)
        for d in range(1, max_d+1):
            nei = pos - d
            oldMask = validOffsetsMask[nei]
            if oldMask == 0:
                continue
            # Now we want to see which offsets in 'nei' are still supported,
            # but the adjacency structure is stored from nei-> (nei+d).  That means:
            #   p = nei, i = p+d = nei+d = pos
            # so d is the same.  We'll read adj[nei][d-1].
            # For offset o' in [0..M-1] at 'nei', it is supported if
            #   adj[nei][d-1][o'] & validOffsetsMask[pos] != 0
            # We'll do a union approach again.
            # But it might be simpler to do a direct check:
            #   newMaskOf(nei) = sum_of o' in oldMask if (adj[nei][d-1][o'] & baseMask) != 0
            row = adj[nei][d-1]  # row of length M: row[o'] is bitmask of valid offsets in pos
            newMask = 0
            bm = oldMask
            while bm:
                low = (bm & -bm)
                o_prime = (low).bit_length() - 1
                bm ^= low
                # check if row[o_prime] & baseMask != 0
                if (row[o_prime] & baseMask) != 0:
                    newMask |= (1<<o_prime)
            if newMask != oldMask:
                removed = oldMask & ~newMask
                validOffsetsMask[nei] = newMask
                if newMask == 0:
                    return False
                Q.append(nei)
        
        return True
    
    # Now we run the queue until empty or conflict.
    while Q:
        i = Q.popleft()
        if validOffsetsMask[i] == 0:
            print("No")
            return
        if not prune(i):
            print("No")
            return
    
    # If we exit with no empties, that means we have a consistent assignment
    print("Yes")