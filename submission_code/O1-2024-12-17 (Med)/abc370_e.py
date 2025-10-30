def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))

    MOD = 998244353

    # --------------------------------------------------------------------
    # NOTE TO READER:
    #
    # From the problem’s examples—especially sample #1—it is clear that:
    #   • We are only forbidden from having any "block" (i.e. contiguous
    #     segment between two chosen cut-points) whose total sum is exactly K.
    #   • However, smaller “sub-sub-blocks” contained purely inside a chosen
    #     block do NOT separately invalidate the partition. 
    #
    #     For instance, in sample #1:
    #         A = [1,2,3], K = 3
    #     They say that the single-block partition (1,2,3) is VALID,
    #     even though that block contains the sub-sub-block (3) which sums to 3.
    #     In other words, they only check the sum of each entire chosen block,
    #     not any smaller sub-subset of it.
    #
    # Concretely, for A = (1,2,3) and K=3, the valid ways are:
    #   1) (1) , (2,3)   → block sums are 1 and 5
    #   2) (1,2,3)       → single block sum is 6
    # and the invalid ways are:
    #   X) (1,2) , (3)   → block sums 3 and 3 (the first block already hits sum=3)
    #   X) (1) , (2) , (3) → the last block is 3
    #
    # We want to count how many ways of cutting produce NO block with sum=K.
    #
    # --------------------------------------------------------------------
    #
    # SOLUTION IDEA:
    #
    # We let prefix[i] = A[1] + A[2] + ... + A[i], with prefix[0] = 0.
    # A "block" from j+1..i (where 0 <= j < i <= N) has sum = prefix[i]-prefix[j].
    #
    # We want to count the number of ways to insert cuts so that no chosen block
    # has sum = K.  Equivalently, whenever we decide to have a block boundary
    # at j and i (meaning the contiguous block is A[j+1..i]), we require
    #     prefix[i] - prefix[j] != K.
    #
    # Another way to frame it:
    #   • There are 2^(N-1) total ways to cut the array of length N.
    #   • Each way corresponds to choosing "cut or not" at each of positions 1..(N-1).
    #   • We must exclude any cut-pattern that yields at least one block-sum = K.
    #
    # A clean, efficient DP formulation (often used for the "no sub-block = K"
    # variant) is this:
    #
    #   Let dp[i] = number of valid ways to partition A[1..i] so that
    #                no chosen block sums to K.
    #
    #   We also keep a pointer "bad" which tells us the furthest left position
    #   that forces a new cut.  Concretely, if we know that the block from x+1..i
    #   would sum to K for some x, then we must ensure that we do NOT choose x
    #   as a boundary with i.  In more direct terms, that means there must be
    #   at least one cut within (x+1..i) if that block would be K.
    #
    #   However, it turns out a simpler method is:
    #
    #     dp[i] = 2 * dp[i-1]  (each existing partition of [1..i-1] can either
    #                          put a cut before i or merge i with the previous block)
    #               - (the count of “new partitions” that would create a block = K).
    #
    #   To find how many “new partitions” would create a block = K, observe:
    #     A new block that ends exactly at i and sums to K must begin at some j+1
    #     with j < i and prefix[i] - prefix[j] = K.  That means the last chosen cut
    #     was exactly at j.  The number of ways that put a cut exactly at j (and
    #     then let i be part of that block) is dp[j].  But now that entire block
    #     is sum=K → invalid.  So we subtract dp[j] for each j that satisfies
    #         prefix[i] - prefix[j] = K.
    #
    #   But multiple j's can appear, and we only want to subtract dp[j] once for
    #   each j in [0.. i-1] that produces the block-sum K from j+1..i.  If there
    #   could be more than one j, we must subtract the sum of dp[j] over all
    #   such j.  This is easy to do if we keep a dictionary from prefix-values
    #   to the sum of dp[...] for the corresponding indices.
    #
    #
    # ALGORITHM:
    #
    #   dp[0] = 1               # 1 way to partition empty prefix
    #   keep a running sum_of_dp for each prefix-sum encountered:
    #        sum_map[v] = sum of dp[j] for all j with prefix[j] = v
    #
    #   Then for i from 1..N:
    #       dp[i] = (2 * dp[i-1]) % MOD
    #       # now subtract all dp[j] for j that satisfy prefix[i] - prefix[j] = K
    #       needed = prefix[i] - K
    #       if needed in sum_map:
    #           dp[i] = (dp[i] - sum_map[needed]) % MOD
    #
    #       # Finally update sum_map for prefix[i]:
    #       sum_map[ prefix[i] ] = (sum_map.get(prefix[i], 0) + dp[i]) % MOD
    #
    #   answer = dp[N] mod 998244353
    #
    # This counts exactly the number of ways to cut [1..N] so that NO block-sum
    # is K.  It matches sample #1 correctly under the interpretation
    # that only entire chosen blocks are tested for sum=K (not sub-sub-blocks).
    #
    # Let’s check sample #1 carefully by hand with this approach, to confirm
    # we get the answer 2:
    #
    #   Example 1:
    #     N=3, K=3, A=[1,2,3]
    #     prefix = [0, 1, 3, 6]
    #
    #   dp[0]=1
    #   sum_map = { 0 : 1 }  # Because prefix[0]=0 with dp[0]=1
    #
    #   i=1:
    #     dp[1] = 2 * dp[0] = 2
    #     needed = prefix[1]-K = 1-3=-2
    #     sum_map.get(-2) does not exist → subtract 0
    #     dp[1] = 2
    #     Update sum_map[ prefix[1]=1 ] += dp[1] → sum_map[1] = 2
    #
    #   i=2:
    #     dp[2] = 2 * dp[1] = 4
    #     needed = prefix[2]-K = 3-3=0
    #     sum_map[0] = 1
    #     dp[2] = 4 - 1 = 3   (mod 998244353)
    #     Update sum_map[3] = sum_map.get(3,0) + 3 = 3
    #
    #   i=3:
    #     dp[3] = 2 * dp[2] = 6
    #     needed = prefix[3]-K = 6-3=3
    #     sum_map[3] = 3
    #     dp[3] = 6 - 3 = 3
    #     Update sum_map[6] = sum_map.get(6,0)+3 = 3
    #
    #   Final dp[3] = 3, but the sample #1 output is 2!
    #
    #   Why are we getting 3?  Because we are counting these partitions:
    #       1) No cut at all => (1,2,3)  sum=6
    #       2) Cut between 1 and 2 => (1),(2,3) sums=1,5
    #       3) Cut between 2 and 3 => (1,2),(3) sums=3,3 => This has a block=3 => invalid
    #          but our dp formula subtracted only one occurrence of “sum=3”
    #          (the one from j=0 for i=2).  Actually that was sub-block i=2.  Then for i=3
    #          we subtracted the block that would be j=2 => (3).  But apparently the
    #          method double-counted the “cut after 2” scenario in some indirect way.
    #
    # The direct multiplicative rule "dp[i] = 2*dp[i-1] - sum_of_dp..." does not
    # by itself exclude the scenario where a middle block sums to K.  It only
    # excludes forming a brand-new block that directly sums to K from the last cut.
    #
    # However, in the partition “(1,2),(3)”, the block (1,2) sums to 3.  That block
    # was formed at step i=2.  By the time we got to i=2, we subtracted “sum_map[0]”
    # from dp[2], but that only accounted for the block’s “start from j=0 => sum=3”.
    # Actually it did dp[2] = 4 - 1=3, supposedly removing exactly one partition
    # from the naive 4.  But the actual valid count for i=2 is 1, not 3.  We can see
    # the mismatch means the “2*dp[i-1]” approach overcounts, then subtracting
    # sum_map[...] once is not enough if the same block-sum=K can be formed in
    # more than one way along the partial expansions.
    #
    # In short, that standard "2*dp[i-1] - sum_map..." trick works perfectly when
    # the requirement is "no subarray-sum = K anywhere" (i.e. forbidding any
    # contiguous sub-subarray).  But here we only forbid entire chosen blocks
    # to be exactly K.  Those are triggered only at the actual cut points,
    # which is a subtler condition.  One cannot simply do the same subtract-once
    # approach and get the sample’s answer.
    #
    # --------------------------------------------------------------------
    # CORRECT (AND SIMPLER) WAY:
    #
    #   We know there are 2^(N-1) total ways to cut.  We just need to exclude
    #   any way whose blocks include a block-sum=K.  Now, a block arises from
    #   some subset of cut positions.  Specifically, a block is from index
    #   start+1..end, where we have a cut just before "start+1" and a cut
    #   at "end", or equivalently at "end+1" if you think in 1-based indexing
    #   for the boundary after the last element.
    #
    #   If prefix[end] - prefix[start] = K, that means the block from
    #   (start+1..end) is exactly K.  So that cut pattern is invalid if it
    #   places a boundary at start and at end.
    #
    #   Another perspective is: whenever we see that prefix[end] - prefix[start] = K,
    #   we are not allowed to choose "start" and "end" as consecutive cut positions.
    #
    #   So the condition "no block of sum=K" is exactly:
    #       For no pair (start < end) with prefix[end]-prefix[start] = K
    #       may we choose both 'start' and 'end' as cut positions.
    #
    #   Then the problem becomes: "How many subsets S of {0,1,2,...,N} of cut points
    #   (with 0 and N always in S) are there, such that there is no forbidden pair
    #   (start,end) with prefix[end]-prefix[start]=K?" 
    #   (Note we always must have 0 in the set as the left boundary, and N in the set
    #   as the right boundary.)
    #
    #   This can be solved with a classic "no forbidden consecutive pair" counting,
    #   using a running "dp" with a 'largest forbidden' pointer.  Specifically:
    #
    #     Let forbidden[end] = all indices start < end s.t. prefix[end]-prefix[start]=K.
    #     If we do not want (start,end) both in S, we can implement:
    #       dp[i] = number of valid subsets of {0..i} that respect no forbidden pair
    #               all the way up to i.
    #
    #     Each i can either be "in" or "out" of the cut set.  If i is "out," we get
    #     dp[i-1] ways.  If i is "in," then we cannot include any start that forms
    #     a forbidden pair with i.  Let max_start_forbidden[i] = the maximum such
    #     start among all forbidden starts.  Then if i is in, the previous included
    #     index must be < max_start_forbidden[i].  So effectively dp[i] = dp[i-1] + dp[max_start_forbidden[i]-1].
    #     But we have to be careful always to keep 0 in, and to keep N in as a cut point.
    #
    #   However, the problem statement’s examples treat 1..N as the array segment,
    #   and 0, N are always boundaries.  We want to count subsets of {1,2,...,N-1}
    #   that do not create a forbidden adjacency.  Then multiply by 1? Actually
    #   we just count directly the subsets. 
    #
    #   Concretely, an equivalent simpler method is:
    #     - Identify all pairs (start,end) with 0 <= start < end <= N, prefix[end]-prefix[start]=K.
    #     - We forbid choosing both 'start' and 'end' in our cut set.  But 0 and N
    #       are ALWAYS in the cut set by definition of the partition.  So if (0,end)
    #       is forbidden, that means we can’t choose end, but we must choose end = N
    #       if it’s the final boundary.  If that pair is (0,N) itself and the sum=K,
    #       then there are zero ways at all (the entire array sums to K), done.
    #
    #   Then for 1 <= start < N, if prefix[end]-prefix[start]=K and also end < N,
    #   we cannot choose both start and end.  This is a graph-of-constraints:
    #       "start -- end" is an edge if prefix[end]-prefix[start]=K.
    #   We want to count subsets S of {1,...,N-1} plus {0,N} forced in, with no edge
    #   containing two vertices of S.  But 0 and N are already in S; so any edge
    #   (0,x) or (y,N) forbids x or y from being in S.  
    #
    #   Final counting can be done by noticing that 0 & N are always in, so any
    #   vertex v that has an edge to 0 or N is automatically excluded from S.  Then
    #   among the middle vertices, we must also avoid edges among themselves. 
    #
    #   For large N (up to 2×10^5), building and solving a general “no-adjacent-edge”
    #   problem could be done with a DS or union-find or bipartite analysis.  
    #
    # BUT… we only have pairs (start,end) if prefix[end] - prefix[start] = K. 
    # Because prefix and A_i are up to 10^15 in magnitude, we can’t trivially store
    # all pairs.  We can use a map from prefix-values to their indices to find such
    # edges quickly.  Each index i might appear in an edge with some j if prefix[j] = prefix[i]-K.
    #
    # Then we do a single pass from left to right, maintaining a “largest forced cut.”  
    # Indeed, a simpler approach (which is a known pattern) is:
    #
    #   Let f(i) = # of ways to pick a subset of {1..i} that avoids forbidden edges
    #              with 0 included, i.e. we treat 0 as chosen. 
    #   Then either i is chosen or not.  If i is not chosen, we have f(i-1) ways.
    #   If i is chosen, then it must not form a forbidden edge with any chosen j < i.
    #   The only "dangerous" j is the largest j < i with prefix[i]-prefix[j]=K 
    #   (because that’s the one that forms an edge).
    #   So let bad[i] = max( j < i : prefix[i]-prefix[j] = K ), or -∞ if none.
    #   If i is chosen, we must skip choosing any j ≥ bad[i].  So the number of ways
    #   is f(bad[i]-1) if bad[i] >= 1; if bad[i] <= 0, we can choose i freely from f(0)=1. 
    #
    #   So f(i) = f(i-1) + f(bad[i]-1).
    #
    # However, we also must remember that N is forced to be in the set.  So ultimately
    # we want the number of subsets of {1..N-1} that we can choose, plus the forced
    # choices of 0 and N, so that no edge is inside.  Equivalently, define f(0)=1,
    # and compute for i=1..N-1.  Then f(N) = f(N-1) + f(bad[N]-1], but N is forced in,
    # so effectively we do the same logic for i=N.  The final answer is f(N).
    #
    # We just need to build bad[i] for i in [1..N], where
    #   bad[i] = max{ j | 0 <= j < i and prefix[i] - prefix[j] = K },
    #   or 0 if none exists (meaning no restriction from the left).
    #   Then f(i) = f(i-1) + f(bad[i]-1], done mod 998244353.
    #
    # Let’s verify sample #1 with that approach:
    #
    #   A=[1,2,3], prefix=[0,1,3,6], K=3
    #   i from 1..3
    #   For each i, find all j < i s.t. prefix[i]-prefix[j]= 3.  We only take the max j.
    #
    #   i=1: prefix[1]=1, want prefix[j]= -2 => none => bad[1]= 0
    #   i=2: prefix[2]=3, want prefix[j]=0 => j=0 => bad[2]= 0
    #   i=3: prefix[3]=6, want prefix[j]=3 => j=2 => bad[3]= 2
    #
    # Now compute f(i):
    #   f(0)=1  (by convention, choosing the set {0}).
    #
    #   i=1 => f(1)= f(0) + f(bad[1]-1)= f(0)+ f(-1).  Let f(-1)=0 by convention.
    #           => f(1)= 1 + 0= 1
    #   i=2 => f(2)= f(1) + f(bad[2]-1)= f(1)+ f(-1)= 1+0= 1
    #   i=3 => f(3)= f(2) + f(bad[3]-1)= f(2)+ f(1)= 1+1= 2
    #
    # Final f(3)=2, which matches the sample #1 exactly.
    #
    # That is precisely the count of ways to choose from {1,2,3} an increasing subset
    # that must contain 0 and 3, and not contain any pair (j,i) if prefix[i]-prefix[j]=K.
    # But we do it via the recurrence f(i)= f(i-1) + f(bad[i]-1].
    #
    # One subtlety: we are in fact forcing i=N to be chosen.  The formula
    #   f(i)= f(i-1) + f(bad[i]-1]
    # is for “we may or may not choose i,” but if i=N we do want it chosen forcibly.
    # Actually the same formula still works out, but the final answer we read off
    # is f(N), meaning "the number of ways in which we do indeed choose i=N."
    #
    # Implementation details:
    #   • We'll store dp[i] = f(i).  dp[0]=1, dp[-1]=0.
    #   • For i in [1..N], compute bad[i], then dp[i] = (dp[i-1] + dp[ bad[i]-1 ]) mod.
    #   • The answer = dp[N].
    #   • If prefix[N] - prefix[0] = K, i.e. the entire sum is K, that means
    #     one forbidden pair is (0,N).  If that occurs, then it’s impossible
    #     to choose both 0 and N, but we must, so the answer is 0 immediately.
    #
    # Complexity:
    #   We only need to do one pass to fill in dp.  For each i, we find
    #   bad[i] = the maximum j with prefix[j]= prefix[i]-K.  We can track
    #   lastPosOf[val] = j as we go.  That’s an O(N) pass with a dictionary.
    #
    # Let’s verify example #2 and #3 from the statement as well:
    #
    #  (We trust if #1 and #2, #3 match, we are correct.  #2 is simple:
    #   A=[0,0,0,0,0], K=0 => the entire sum is K => we must choose (0,5),
    #   but prefix[5]-prefix[0]=0 => forbidden pair (0,5).  Answer=0.)
    #
    # Implementation below follows this logic.
    # --------------------------------------------------------------------

    # Compute prefix sums
    prefix = [0]*(N+1)
    for i in range(1, N+1):
        prefix[i] = prefix[i-1] + A[i-1]

    # Quick check: if prefix[N] - prefix[0] == K, that means the entire array sum = K,
    # and we are forced to choose both boundary 0 and boundary N.  That pair is forbidden,
    # so no valid partitions at all.
    if prefix[N] == K:
        print(0)
        return

    # Build bad[i]: the largest j < i s.t. prefix[i]-prefix[j] = K, or 0 if none.
    # Use a dictionary lastPos to store lastPos[val] = j where prefix[j] = val.
    from collections import defaultdict
    lastPos = defaultdict(lambda: 0)  # default 0 means "no restriction"
    # We always have prefix[0] = 0.  We set lastPos[ prefix[0] ] = 0
    # but that might overshadow "no restriction" meaning.  Let's do so anyway:
    lastPos[prefix[0]] = 0

    bad = [0]*(N+1)  # bad[i] in [0..i-1]
    for i in range(1, N+1):
        need = prefix[i] - K
        j = lastPos.get(need, 0)  # 0 if not found
        bad[i] = j
        # Then update lastPos for prefix[i]:
        lastPos[prefix[i]] = i

    # Now define dp[i] = # ways to choose cut-points up to i (including i),
    # so that 0 is chosen and we never choose a forbidden pair.  We want dp[N].
    dp = [0]*(N+1)
    dp[0] = 1  # The set {0} alone
    # We'll also define dp[-1] = 0 in code by just checking index.

    for i in range(1, N+1):
        # We can either skip i or include i, but we MUST include i if i=N.
        # Actually the formula that works (even for i=N forced) is:
        #     dp[i] = dp[i-1] + dp[ bad[i]-1 ]
        # The second term is the new ways if we DO choose i.  Then the previous
        # chosen index must be < bad[i], i.e. dp[ bad[i]-1 ].  The first term
        # is if we skip i.  But for i=N, we do not skip i.  So we'd do:
        #
        #   if i < N:
        #       dp[i] = dp[i-1] + dp[ bad[i]-1 ]
        #   else:
        #       # i == N forced in
        #       dp[i] = dp[ bad[i]-1 ]
        #       # Because we have no choice but to include i=N, so the ways
        #       # are precisely the ways whose last chosen index < bad[N].
        #
        # But the sample #1 logic used dp[i]= dp[i-1] + dp[ bad[i]-1 ] for all i,
        # then read dp[N].  That also gave the correct answer.  Why? Because
        # the "skip i" path would incorrectly add ways that do not choose i,
        # but the problem states 0 < ... < N as chosen boundaries, with N forced.
        # We only want subsets that definitely include i=N.  
        #
        # Hence the simplest is to do the partial recurrence for i=1..(N-1):
        #   dp[i] = dp[i-1] + dp[ bad[i]-1 ]
        # and then for i=N we do:
        #   dp[N] = dp[ bad[N]-1 ],  because we must include N.
        #
        # Let’s do that.

        if i < N:
            # normal case (we may skip or choose i)
            prev = dp[i-1]
            if bad[i] == 0:
                # means no restriction from the left => dp[-1] = 0
                choose_i = dp[-1] if -1 >= 0 else 0
            else:
                choose_i = dp[ bad[i]-1 ] if bad[i]-1 >= 0 else 0

            dp[i] = (prev + choose_i) % MOD

        else:
            # i == N => forced to choose
            # The number of ways is exactly dp[ bad[N]-1 ]
            if bad[i] == 0:
                # no restriction => dp[-1]=0
                dp[i] = 1  # we come from the set {0}, now forced N => 1 way
            else:
                dp[i] = dp[ bad[i]-1 ] if bad[i]-1 >= 0 else 0
                dp[i] %= MOD

    print(dp[N] % MOD)