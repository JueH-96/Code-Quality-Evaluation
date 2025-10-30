class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        """
        We want to transform an even-length array nums into some final array
        such that there is a single integer X where for every i,

            abs(final[i] - final[n-1-i]) = X.

        We can change any element of nums to a value in [0..k]. Our goal is to
        minimize the number of changes.

        -----------------------------------------------------------------------
        OBSERVATION AND APPROACH:

        We pair up elements (i, n-1-i). Each pair (a, b) must satisfy:
            abs(a', b') = X    (where a', b' are the possibly changed values)

        We want a single X to work for all pairs, and we want to minimize total
        changes.

        Let cost_{(a,b)}(x) be how many changes needed so that abs(a', b') = x
        with a', b' in [0..k]. This cost is in {0,1,2}:

        - cost = 0 if abs(a - b) == x (already done; no change).
        - cost = 1 if we can fix one element and change the other to get x,
          without requiring two changes. That is, either:
            abs(a - b') = x for some b' in [0..k] with b' != b, or
            abs(a' - b) = x for some a' in [0..k] with a' != a.
          If the change does not effectively match the cost=0 condition, it
          remains cost=1.
        - cost = 2 otherwise.

        The key to an efficient solution is to avoid enumerating all x in [0..k]
        for each pair (which can be too large). We note that for each pair (a, b),
        the only x values that can yield cost < 2 must come from a small set
        derived from (a, b).

        Specifically,
          cost=0  => x0 = abs(a - b).
          cost=1  => x must be such that it can be formed by changing exactly one
                     element to a value in [0..k]. That leads to at most 4 new x
                     values: abs(a - (a±x)), abs((b±x) - b). Concretely:
             - b' = a + x  or  b' = a - x (if within [0..k])
             - a' = b + x  or  a' = b - x (if within [0..k])
            Each valid move in range introduces a candidate difference.

        So for each pair, we collect these candidate differences x, discerning
        whether the cost for that x is 0 or 1. All other x in [0..k] will have cost=2.

        Then, to combine over pairs, we do:

          sumCost[x] = sum_{over all pairs} cost_{(a,b)}(x).

        We initialize sumCost[x] to be 2 for each pair by default (the worst case),
        i.e. total contribution of 2*n/2 = n. Then for each pair, we "reduce" the cost
        from 2 down to the actual cost(0 or 1) for those x that the pair can achieve
        cheaper. Finally, we pick the minimal sumCost[x] over x in [0..k].

        The complexity is O(n) to iterate over the pairs, each pair contributes up to
        5 distinct x values (the cost=0 difference abs(a-b), plus up to 4 from cost=1).
        This is efficient for n up to 10^5 and k up to 10^5.

        -----------------------------------------------------------------------
        Example:
        nums = [1,0,1,2,4,3], k = 4
        We have 3 pairs: (1,3), (0,4), (1,2).
        We'll compute for each pair which x values lead to cost=0 or 1, and sum up.

        We then pick the x with minimal overall cost.

        -----------------------------------------------------------------------
        Implementation below:
        """
        n = len(nums)
        half = n // 2

        # We'll store total cost in sumCost[x]: sum of cost_{(a,b)}(x) across all pairs.
        # By default, everything is 2 changes per pair => total 2*half.
        sumCost = [2 * half] * (k + 1)

        for i in range(half):
            a = nums[i]
            b = nums[n - 1 - i]

            # Gather cost<2 candidates in a small map: diff_map[x] = cost
            diff_map = {}

            # cost=0 for x0 = abs(a-b)
            x0 = abs(a - b)
            diff_map[x0] = 0 if x0 <= k else 2  # only if x0 <= k is it a valid difference in [0..k]

            # Possible b' = (a ± someX) in [0..k], cost might be 0 (if b' = b & x = x0) or 1 otherwise
            # We'll just try b' = a+x, b' = a-x as valid changes:
            #   x = abs(a - b') => see if that x <= k
            #   cost = 1 unless it duplicates cost=0 scenario.
            candidates_bprime = []
            # b' = a + something means b' in [0..k], so something = b' - a
            # We'll skip enumerating all, we just check a+x, a-x for x≥0
            if a + b <= k:  # This is not correct usage. We want to vary x. Instead:
                pass
            # Instead do direct checks:
            # b' = a + x. Then x = |a - (a + x)| = x. For that to land in [0..k], we need 0<= a + x <= k.
            # But we are inside a loop over pairs, not x. We want to see which x would arise when b' in {a+x, a-x}.
            # We'll compute each b' and then the difference xCand = abs(a - b')
            # Then see if 0 <= xCand <= k. That xCand is a candidate for cost=1 if b' != b or it isn't already cost=0.

            # b' = a + something
            # We'll directly form b' in {a+x, a-x} and see if 0 <= b' <= k
            # Then let xCand = abs(a - b')
            # Mark cost=1 if it's not cost=0 and we are changing b' from b (unless it's the same difference). See below.
            possible_bprimes = []
            up_b = a + b  # This is not correct. We want a + something that yields cost=1. Let's do it systematically:

            # We'll check b' in {a + x0, a - x0}, but actually that loops back to cost=0 if b'==b.
            # But for cost=1 we want all xCand = abs(a - b') with b' in [0..k], b' != b OR if xCand = x0 we might get cost=0.
            # However, we only need to consider exactly 2 values for b': a+b, a-b doesn't make sense, we want a ± X. 
            # Actually, we want to handle all unique differences we can get by changing b or a once.

            # Let's define a function that tries changing b:
            def try_change_b(bp):
                """Check the difference xCand = abs(a - bp). Mark cost=1 if valid."""
                if 0 <= bp <= k:
                    xCand = abs(a - bp)
                    # If xCand <= k, it is a valid difference
                    if xCand <= k:
                        # Determine cost:
                        # cost=0 if xCand == x0 and bp == b => already no change needed
                        # otherwise cost=1 if we changed b (unless it coincidentally also means x= x0 without changing?)
                        if xCand == x0 and bp == b:
                            c = 0
                        else:
                            c = 1
                        if xCand in diff_map:
                            diff_map[xCand] = min(diff_map[xCand], c)
                        else:
                            diff_map[xCand] = c

            # Similarly, try changing a:
            def try_change_a(ap):
                """Check the difference xCand = abs(ap - b). Mark cost=1 if valid."""
                if 0 <= ap <= k:
                    xCand = abs(ap - b)
                    if xCand <= k:
                        # cost=0 if xCand == x0 and ap == a => no change
                        if xCand == x0 and ap == a:
                            c = 0
                        else:
                            c = 1
                        if xCand in diff_map:
                            diff_map[xCand] = min(diff_map[xCand], c)
                        else:
                            diff_map[xCand] = c

            # Now, for cost=1, we consider b-> a ± something but "something" = xCand.
            # We'll do a direct approach: we'll check b' = a+1, a-1, ..., but that is too large in general.
            # Instead, we know that changing b by one operation can yield differences:
            #   xCand = abs(a - b') for some b' in [0..k].
            #   But to find all xCand that are "one-change" away, we only need b' in {a+x0, a-x0, etc...}?
            #
            # Actually, the easiest correct approach is:
            #   We can produce up to 2 candidate b' from the condition "abs(a - b') = x" for each x,
            #   but that's still enumerating x. Instead, let's systematically try all b' in {a±1, a±2,...} is too large.
            #
            # More efficient approach: The only relevant b' that might yield cost=1 (and not cost=0) are:
            #   b' in {a + d, a - d} for d small enough, but we don't want to iterate all d...
            #
            # In practice, we only need to test the two possible ways to produce difference x0 from 0 changes,
            # plus the two ways each for a single change. That is exactly 4 tries: b' = a+b - b ??? This is confusing.
            #
            # The standard known approach is:
            #   For cost=1 from changing b alone: The difference xCandidate = abs(a - (some b')), for some b' in [0..k].
            #   The possible b' that is at distance (b' != b) can't be enumerated in a broad sense for large k.
            #   But note: if the difference is xCandidate, then b' = a±xCandidate. So let's systematically do:
            #       b' = a + 0, a - 0, a + 1, a - 1, ..., a + k, a - k
            #   That is still O(k). This is not feasible for large k directly. 
            #
            # However, we only truly need to fill cost<2 differences in `diff_map`. The rest remain cost=2.
            # That set is: {abs(a-b), abs(a - (a+x)), abs(a - (a-x)), abs((b+x) - b), abs((b-x) - b)} for x in [0..k].
            # But again that references "x in [0..k]" which is too big.
            #
            # A more direct method is: cost=1 means exactly one element changes. So let's collect the possible
            # differences we can get by either "a stays the same, b changes" or "b stays the same, a changes".
            #
            # If we keep 'a', then any difference x = abs(a - b') is possible if b' in [0..k].
            # In fact, for any x, b' = a±x. So for cost=1, we get x from {abs(a - y) | y in [0..k], y != b} but that's
            # too large if we iterate y in [0..k].
            #
            # But to build the final sumCost array, we do not need to generate all x. Instead, we do:
            #   sumCost[x] starts at 2 * (#pairs).
            #   For each pair, if cost_{(a,b)}(x) < 2, we reduce sumCost[x].
            # That again suggests we must visit all x, which is too big.
            #
            # The well-known technique is that for a single pair (a,b), the set of x for cost < 2 is of small size:
            #   1) x0 = abs(a-b) => cost=0
            #   2) Up to 4 more x for cost=1, from the conditions:
            #      - change b => b' in {a + something, a - something} but that "something" is exactly b - a or something else?
            #      - Actually, there's a standard "trick": cost=1 means either (a' = b ± x0) or (b' = a ± x0) for the difference to end up x0?

            # Let's do the proven small-check approach:
            #
            # cost=1 arises if exactly one side changes. Then the resulting difference is one of:
            #   abs(a - b'), b' in [0..k], or abs(a' - b), a' in [0..k].
            # Among these, the only possible "b'" that can yield a difference are b' = a + d or b' = a - d for some d.
            # But we do not know d. 
            #
            # However, we can note: if b' = a + d, then the difference is d. If b' = a - d, difference is d too. 
            # So effectively, if we can pick b' = x + a, that difference is x. We only need to check if b' is in [0..k].
            #
            # So for difference x to be possible by changing b, we need b' = a + x or b' = a - x to be in [0..k]. If b' = b
            # we revert to cost=0 if that also matches x=abs(a-b). Otherwise cost=1. 
            #
            # Similarly, for difference x by changing a, we need a' = b + x or a' = b - x in [0..k]. 
            #
            # Summarizing (this is the standard approach):
            #   Let x0 = abs(a-b). => cost=0
            #   For cost=1, the set of x includes any x for which:
            #       (a + x) in [0..k], (a + x) != b => cost=1 or 0 if x= x0 and b = a+x
            #       (a - x) in [0..k], (a - x) != b => cost=1 or 0 if x= x0 and b = a-x
            #       (b + x) in [0..k], (b + x) != a => cost=1 or 0 if x= x0 and a = b+x
            #       (b - x) in [0..k], (b - x) != a => cost=1 or 0 if x= x0 and a = b-x
            #
            # But we don't want to loop x in [0..k]. We want a small set of x. The trick is that for each pair,
            # the possible x that yield cost<2 are precisely:
            #   x0 = abs(a - b) for cost=0
            #   x1 in {abs(a - (a+x)), ...} => this is effectively x = any nonnegative integer up to k
            # This looks large, but let's do it in the correct simpler code form:

            # We'll explicitly gather the pairs (a', b) or (a, b') for which cost=1 might happen:
            # For "change b" candidate b' = a±(some integer). That integer is b' - a. But we only want to check b' in {b±x0}?
            # This also doesn't seem correct. 
            #
            # The known standard solution is (as typically used in editorial approaches):
            #   - Start sumCost[x] = 2 * (#pairs).
            #   - For each pair, you do:
            #       sumCost[abs(a-b)] -= 2  # because cost=0 => 2 -> 0 is a difference of 2
            #       For the up to 4 possible x from cost=1, you do sumCost[x] -= 1. 
            #
            # And the up to 4 possible x are found by checking b' = a±1, a±2, etc. is again too many if we do it literally.
            #
            # But there's an actual known formula for these up to 4 x for cost=1:
            #   Let xCandidate1 = abs(a - (b+1)) if (b+1 <= k), xCandidate2 = abs(a - (b-1)) if (b-1 >= 0),
            #   xCandidate3 = abs((a+1) - b) if (a+1 <= k), xCandidate4 = abs((a-1) - b) if (a-1 >= 0).
            #
            # Intuition: changing b by ±1 might yield a difference that is close, but what if we want to add 2? 
            # We only consider ±1 because we want to see if we can get difference x with exactly one change? That single
            # change could move b from b to anything in [0..k], not just b±1. So ±1 is not enough. We need to consider
            # a -> 0, a -> k, etc. That is large again. 
            #
            # The correct known approach is: for cost=1, we can fix one side, and choose the other side to be "the old side ± x".
            # That is: if we fix a, difference = x => b' = a ± x. If 0 <= a±x <= k, then cost=1 for that x unless it was cost=0 or
            # we didn't actually change b (b' != b?). Similarly if we fix b => a' = b ± x. 
            #
            # So for each pair (a,b), the possible x that yield cost=0 or cost=1 are:
            #    x0 = abs(a-b) => cost=0
            #    Then for cost=1:
            #      x = abs(a - (a±u)) for all u s.t. 0 <= a±u <= k. But that is again many u. 
            #
            # However, we do not want to produce "all x <= k". We only subtract 1 from sumCost[x] for those x that can be achieved
            # by changing exactly one element. But that set is large. The key is a typical trick in "two-pointer" editorial solutions:
            #
            # Instead of enumerating x, we do a prefix-difference approach:
            #  - For difference x in [0..a], we can set b' = a-x if b' >= 0. That yields x = abs(a-(a-x))=x. So for x in [0..a], if a-x >=0 => valid x
            #    Also b' in [0..k], so x≥ a-k => x≥ negative? 
            #  This can be turned into an interval of x. But it can be complicated. 
            #
            # Because the question is purely "Return the minimum number of changes," we can implement the direct method described:
            #   Default cost = 2 -> sumCost[x] = 2*half for all x.
            #   For each pair:
            #       sumCost[abs(a-b)] -= 2   # turning cost 2 into cost 0 for that x
            #       For each x that can be achieved with cost=1 by changing a or b once, do sumCost[x] -= 1.
            #
            # And we do so by "collecting" all x with cost<2 in a small dictionary. Because for cost=1, we only need to consider
            # at most 4 possible new differences:
            #   For b -> a±x to be in [0..k], x = abs(a - b'). So b' can be any integer in [0..k], but we can't iterate all.
            #   There's a known approach: we only consider b' in {0, k}, and similarly a' in {0, k}, plus the original b or a. 
            # That yields at most 5 possible values for b' each (the original b plus the extremes 0, k). Then we collect the differences.
            #
            # Because b' could be anywhere in [0..k], but the cost in the final search only depends on whether there exists some
            # b' that yields difference x with a single change. If there is any b' in [0..k], cost=1 is possible. Checking b'=0, b'=k
            # might not be enough if the needed b' is somewhere else. 
            #
            # A pragmatic editorial solution that works in practice is to store:
            #   ret[abs(a-b)] = 0
            #   for b' in {0, k}:
            #       x = abs(a-b'); cost=1
            #   for a' in {0, k}:
            #       x = abs(a'-b); cost=1
            #
            # plus a check if that x actually equals abs(a-b) and b'==b => cost=0. 
            # But that might fail if the needed b' is e.g. a+5 for some 5 not in {0, k} if that is the only way to get cost=1. 
            # Actually, we can do better by noticing that if a ≤ k, then any difference x in [0..k] can be formed by picking b' = a±x
            # as long as a±x is in [0..k]. So to represent all possible x from changing b we can do:
            #   for x in range(k+1):
            #       if 0 <= a±x <= k => cost=1
            # which is still O(k) per pair => O(nk) => 1e10 in worst case, too big. 
            #
            # The known official solution approach (seen in editorial of similar problems) is:
            #   We'll build a map for each pair that says: ret[x] in {0,1} if cost<2. Then we do a large sweep. But the issue is how
            #   to fill that map efficiently without enumerating x in [0..k].
            #
            # However, there's a simpler observation:
            #   cost_{(a,b)}(x) can be 0 or 1 only if x is in {abs(a-b),
            #   abs(a - 0), abs(a - k), abs(0 - b), abs(k - b)}. Because changing b to 0 or k or changing a to 0 or k often
            #   helps capture extremes. But what if the needed b' was in the middle? Actually, that might be needed.
            #
            # But there's a key insight: if we can choose any integer from 0..k for b', then for a fixed a, the set of possible
            # differences abs(a - b') as b' runs from 0..k is exactly all integers from 0..k if a ≤ k. Indeed, if a ≤ k,
            # you can get differences from 0..max(a, k-a?), effectively up to k. So that alone is the entire range [0..k].
            # That doesn't yield a "small set". 
            #
            # The standard technique is indeed to do a single pass over x from 0..k, summing cost_{(a,b)}(x). But that is O(n*k)=1e10,
            # which is too big in worst case. 
            #
            # There must be a more mathematically direct argument that any pair (a,b) can produce cost=1 for ANY x in [0..k], except
            # if we specifically cannot set a or b to produce that difference with one change. Actually, we can always produce any x <= k
            # with one change if |a-b| != x, except if it's impossible because we can't find a' or b' in [0..k]. 
            # Example: if we want difference x, we can fix a and choose b' = a±x if 0<= a±x <= k, or fix b and choose a'= b±x if 0<= b±x<=k.
            # So cost=1 is possible for x if (a+x <= k or a-x >= 0) or (b+x <= k or b-x >=0), unless it was already cost=0 for that x. 
            #
            # So let's define:
            #   cost_{(a,b)}(x)=0 if x=abs(a-b).
            #   cost_{(a,b)}(x)=1 if x!= abs(a-b) and [ (a+x <= k or a-x>=0) or (b+x<=k or b-x>=0 ) ].
            #   cost_{(a,b)}(x)=2 otherwise.
            #
            # Then to find sumCost[x], we do:
            #   sum_{pairs} cost_{(a,b)}(x).
            # We'll do a "line sweep" approach:
            #   Initialize sumCost[x] = 2 * (#pairs).
            #   For each pair:
            #       Let x0 = abs(a-b).
            #       sumCost[x0] -= 2 (because cost=0 instead of 2).
            #       Then let L = min(a,b), H = max(a,b). We check the intervals in x for which cost=1 is possible:
            #       cost=1 if x != x0 AND (x in the range(s) implied by (a±x in [0..k]) or (b±x in [0..k])).
            #
            # We can break those conditions into intervals on x:
            #   (a+x <= k) => x <= k-a
            #   (a-x >= 0) => x <= a
            #   (b+x <= k) => x <= k-b
            #   (b-x >= 0) => x <= b
            # So the union of intervals is [0..max(a, k-a, b, k-b)] basically, but we still have to exclude x0. 
            # Then for each x in that union, cost=1 => sumCost[x] -= 1 (since the default is 2, we reduce from 2->1).
            #
            # We can merge intervals for "fix a" and "fix b":
            #   For fix a => x in [0.. a] union [0.. k-a] => actually [0.. max(a, k-a)]
            #   For fix b => x in [0.. b] union [0.. k-b]
            # So combined => x in [0.. max(a, b, k-a, k-b)].
            # Let's define hi = max(a, b, k-a, k-b). Then for x <= hi, cost might be 1 if x != x0. For x> hi, cost=2. Or maybe we check carefully?
            #
            # Indeed, cost=1 if x <= maxRange, except if x = x0 (that is cost=0) or if neither "fix a" nor "fix b" can produce that x (which means x>k-a and x>a and x>k-b and x>b). So we define:
            #   aRange = max(k-a, a)
            #   bRange = max(k-b, b)
            #   r = max(aRange, bRange)
            # Then cost=1 for all x in [0..r] \ {x0}. For x>r => cost=2. 
            #
            # Implementation: we do a difference array approach on sumCost. 
            # For each pair:
            #   x0 = abs(a-b)
            #   sumCost[x0] -= 2  (cost=0)
            #   r = max(k-a, a, k-b, b)
            #   Then for x in [0..r], cost=1 => we do sumCost[x] -= 1. But that is O(r) which can be up to 10^5. Doing that for 10^5 pairs => 1e10 again too big. 
            #
            # Instead, we do a prefix sums technique:
            #   We'll do an array "delta" of length k+2, initially zeros. 
            #   For [0..r] we want to do sumCost[x] -= 1 => that means delta[0] -= 1, delta[r+1] += 1 (range update). 
            #   But we also must skip x0 => so we do sumCost[x0] += 1 (to undo the -1) if x0 <= r. 
            #
            # So for each pair:
            #   1) sumCost[x0] -= 2  => can do a separate array fix0[x0]++ then after all pairs we'll do sumCost[x] -= 2*fix0[x].
            #   2) For cost=1 range => delta[0]--, delta[r+1]++ if r+1 <= k
            #      if x0 <= r => we skip it => delta[x0]++ and delta[x0+1]-- to correct that position
            #
            # Then at the end, we build sumCost[x] = 2*half + cumulative sum of delta up to x, minus 2 * fix0[x].
            #
            # Let's do it:

            # We'll compute "r" for this pair:
            r = max(a, k - a, b, k - b)

            # We'll handle cost=0 for x0 if x0 <= k:
            # We'll record it separately in fix0 array.
            # We'll record the range for cost=1 in delta:
            # delta[0] -= 1, delta[r+1] += 1, then if x0 <= r, we add 1 at x0 and -1 at x0+1 to skip it.

            pair_data = (a, b)  # to do after this loop

            # We'll store these in a global structure, then apply them after we read all pairs
            # because we need a big delta array outside the loop. We'll do so.

        # Given the time constraints, a direct formula-based range approach is the standard trick.

        # IMPLEMENTATION OF THE RANGE-UPDATE TRICK:

        half = n // 2
        # We'll keep fix0[x] = number of pairs for which abs(a-b)=x
        fix0 = [0]*(k+1)

        # We'll keep two arrays for the difference updates:
        # delta array for the cost=1 range changes
        delta = [0]*(k+2)  # We'll prefix sum over [0..k], so size k+2 to avoid boundary issues

        for i in range(half):
            a = nums[i]
            b = nums[n - 1 - i]
            x0 = abs(a - b)
            if x0 <= k:
                fix0[x0] += 1

            # r = the largest x for which cost can be 1
            r = max(a, k - a, b, k - b)
            if r > k:
                r = k  # cost difference can't exceed k anyway

            # cost=1 for x in [0..r], except x0 if x0 <= r
            delta[0] -= 1
            if r + 1 <= k:
                delta[r + 1] += 1

            if x0 <= r:
                # We add +1 back at x0, and -1 at x0+1
                delta[x0] += 1
                if x0 + 1 <= k:
                    delta[x0 + 1] -= 1

        # Now build sumCost from these arrays:
        # sumCost[x] = 2*half  (initially)
        # Then subtract 2 * fix0[x] => those become cost=0 for that many pairs
        # Then add cumulative sum of delta up to x => that shifts cost from 2 down by 1 for cost=1 ranges
        # and re-corrects cost=0 positions.

        sumCost = [2*half]*(k+1)
        # prefix sum over delta:
        running = 0
        for x in range(k+1):
            running += delta[x]
            # running is how many times we do "-1" to cost => so sumCost[x] += running
            # but running is negative (like -N) means we reduce cost by N
            sumCost[x] += running

        # Now account for cost=0 adjustments:
        for x0 in range(k+1):
            # Each occurrence of fix0[x0] pairs => we sub an additional 1 from sumCost[x0]
            # because we previously subtracted 1 for cost=1, but cost=0 is actually 1 less change than cost=1.
            # Actually we subtracted 2 from 2 => 0 for cost=0? Wait carefully:
            # We started each x with cost=2 => then the delta array sub 1 => cost=1 for pairs that can do cost=1 or 0.
            # But if a pair is cost=0, we need to remove one more from cost=1 => so total sub=2 from the default 2 => cost=0.
            # So we sub (1 more) for each pair that is truly cost=0 at x0. 
            sumCost[x0] -= fix0[x0]

        # Now sumCost[x] is total cost across pairs. We want the minimum:
        return min(sumCost)