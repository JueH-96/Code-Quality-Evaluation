from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        """
        Return the k-th alternating permutation (lexicographic order) of [1 … n].
        An alternating permutation has the property that adjacent elements never
        share the same parity.  If fewer than k such permutations exist, return [].
        """

        LIMIT = 10**15          # k never exceeds this value
        SAT   = LIMIT + 1       # any value bigger than every possible k

        # ---------------------------------------------------------------------
        # factorials (saturated so they never grow above SAT)
        # we only need up to ceil(n/2)! because odds/even sets are that size
        # ---------------------------------------------------------------------
        half = (n + 1) // 2
        fact = [1] * (half + 1)
        for i in range(1, half + 1):
            v = fact[i - 1] * i
            fact[i] = v if v <= SAT else SAT     # saturation

        # ---------------------------------------------------------------------
        # total number of alternating permutations
        # ---------------------------------------------------------------------
        odds_cnt  = (n + 1) // 2
        evens_cnt = n // 2
        total = fact[odds_cnt] * fact[evens_cnt]
        if total > SAT:
            total = SAT
        if n % 2 == 0:                    # both parity patterns are feasible
            total *= 2
            if total > SAT:
                total = SAT

        if k > total:                     # not enough permutations
            return []

        # ---------------------------------------------------------------------
        # helper: multiplication under saturation
        # ---------------------------------------------------------------------
        def mul_sat(a: int, b: int) -> int:
            if a >= SAT or b >= SAT:
                return SAT
            prod = a * b
            return prod if prod <= SAT else SAT

        # ---------------------------------------------------------------------
        # main un-ranking loop
        # ---------------------------------------------------------------------
        remaining = list(range(1, n + 1))   # numbers still available
        res: List[int] = []                 # answer being built
        start_parity = None                 # parity of position 0 (0=even, 1=odd)
        cur_odds, cur_evens = odds_cnt, evens_cnt

        for pos in range(n):                # build positions 0 … n-1
            chosen = False

            for idx, num in enumerate(remaining):
                p = num & 1                 # 0 even, 1 odd

                # decide pattern parity
                if start_parity is None:            # still at position 0
                    pattern_start = p
                else:
                    pattern_start = start_parity
                    if p != (start_parity ^ (pos & 1)):   # wrong parity for slot
                        continue

                # how many odds/evens remain after taking this candidate?
                rem_odds  = cur_odds  - (1 if p else 0)
                rem_evens = cur_evens - (0 if p else 1)
                if rem_odds < 0 or rem_evens < 0:
                    continue

                # -----------------------------------------------------------------
                # check feasibility: remaining numbers must fit the parity pattern
                # -----------------------------------------------------------------
                rem_len = n - pos - 1
                if rem_len:
                    first_needed_parity = pattern_start ^ ((pos + 1) & 1)
                    if first_needed_parity:     # sequence of length rem_len starts with odd
                        need_odds = (rem_len + 1) // 2
                    else:                       # starts with even
                        need_odds = rem_len // 2
                    need_evens = rem_len - need_odds
                    if rem_odds != need_odds or rem_evens != need_evens:
                        continue

                # -----------------------------------------------------------------
                # count permutations with this prefix
                # -----------------------------------------------------------------
                cnt = mul_sat(fact[rem_odds], fact[rem_evens])

                if k > cnt:            # skip this block of permutations
                    k -= cnt
                    continue

                # choose this number
                res.append(num)
                remaining.pop(idx)
                if p:
                    cur_odds -= 1
                else:
                    cur_evens -= 1
                if start_parity is None:
                    start_parity = p
                chosen = True
                break

            if not chosen:              # defensive – should never happen
                return []

        return res