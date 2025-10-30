class Solution:
    def makeStringGood(self, s: str) -> int:
        from functools import lru_cache
        n = len(s)
        cnt = [0] * 26
        for ch in s:
            cnt[ord(ch) - ord("a")] += 1

        # We are going to try candidate final frequencies (k)
        # It turns out that an optimal k is “close” to one of the counts in s.
        candidates = set()
        candidates.add(1)
        for c in cnt:
            if c > 0:
                candidates.add(c)
                candidates.add(c + 1)
        candidate_ks = sorted(candidates)
        
        best_cost = float("inf")
        
        # For fixed candidate k, we want to decide which letters (among the 26, in order)
        # should be “used” (that is, chosen as a target letter in the final good string)
        # so that each such letter will be turned into one of the m distinct letters in t.
        # If we throw away all original letters and insert them, the baseline cost is (n + m*k).
        # But if we can “match” an original letter to a demanded letter then we can “save”
        # (delete + insert) cost. In fact:
        #   – A “direct match” (the letter is already the same) saves 2 operations.
        #   – Using an “earlier” letter that is exactly one step behind (so that one shift is enough)
        #     saves 1 operation.
        #
        # We define the net benefit for a target letter as:
        #    benefit = [2 * (# direct assignments) + (# bonus assignments)] - k,
        # and our goal is to “collect” as much benefit as possible.
        #
        # In our DP we process letters 0 ('a') to 25 ('z').
        # The state dp(i, r, chosen) represents the maximum total net benefit we can get
        # from letters i ... 25 given that:
        #    r is the number of “bonus” occurrences (from the immediately–previous letter) that are
        #      available for use at the current letter i (they would cost 1 shift each),
        #    chosen is a boolean telling whether we have chosen at least one target letter so far.
        #
        # At the end (i == 26) if no target letter was chosen then our option is to “force” insertion
        # of one target letter (which gives net benefit –k).
        
        for k in candidate_ks:
            @lru_cache(maxsize=None)
            def dp(i, r, chosen):
                # i: index 0 ... 26
                # r: available bonus from previous letter (only letters exactly one behind count as bonus)
                # chosen: boolean (True/False) indicating whether at least one target letter has been chosen.
                if i == 26:
                    return 0 if chosen else -k
                # Option 1: skip letter i – we decide not to use letter i as a target.
                # (Then, all occurrences of letter i remain “unused” and become available as bonus for the next letter.)
                option_skip = dp(i + 1, cnt[i], chosen)
                
                # Option 2: choose letter i as a target letter.
                # Then we “try” to get k copies.
                curr = cnt[i]
                direct = curr if curr < k else k  # we get min(cnt[i], k) direct supplies (cost 0 each).
                deficit = k - direct
                bonus_used = r if r < deficit else deficit  # from the bonus available r (these cost 1 shift each).
                benefit = 2 * direct + bonus_used - k
                # When we use letter i as target, then the occurrences of letter i above k
                # will serve as bonus for the next letter.
                new_r = (curr - k) if curr > k else 0
                option_choose = benefit + dp(i + 1, new_r, True)
                
                return option_choose if option_choose >= option_skip else option_skip

            net_benefit = dp(0, 0, False)
            candidate_cost = n - net_benefit
            if candidate_cost < best_cost:
                best_cost = candidate_cost
            dp.cache_clear()
            
        return best_cost