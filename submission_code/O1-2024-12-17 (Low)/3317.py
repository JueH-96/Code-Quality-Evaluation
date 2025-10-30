class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        """
        We can swap characters arbitrarily many times among all words. In effect,
        we have a "bag" of characters and merely need to decide how to distribute
        them back into the words so as to maximize how many of the words become
        palindromes.

        Key observation:
         - Each word of length L requires floor(L/2) "pairs" of characters (each pair
           is the same character repeated twice) to form a palindrome, plus
           (L mod 2) leftover single character if L is odd.
         - Across all words combined, we count total pairs we can form from the global
           multiset of characters, and how many leftover singles we can place.

        Let freq[c] = frequency of character c in the entire input.
          total_pairs = sum(freq[c] // 2 for c in 'a'..'z')
          leftover_singles = sum(freq[c] % 2  for c in 'a'..'z')

        A word with length L:
          p_i = L // 2   (how many pairs needed)
          s_i = L % 2    (how many singles needed: 0 or 1)

        We want the largest subset of words whose combined pairs-needed is <= total_pairs
        and whose combined singles-needed is <= leftover_singles.

        Strategy:
         1) Split words by their singles-needed (s_i = 0 or 1).
         2) Sort each group by p_i ascending.
         3) We will pick up to leftover_singles words from the s_i=1 group (since each
            odd-length palindrome consumes exactly 1 leftover single).
         4) For each possible count x of odd-length words we decide to take first (smallest p_i),
            we then see how many even-length words we can fit, then possibly pick more odd-length
            words if we still have singles left. We track the maximum total.

        This works because:
         - Choosing which odd-length words to include is constrained by leftover singles.
         - Picking them in ascending order of p_i "uses up" the fewest pairs, leaving more
           pairs for other words (either even- or odd-length).
         - We then greedily fill even-length words. Finally, if we still have resources
           (pairs and leftover singles), we can pick more odd-length words from where
           we left off.

        Time complexity is O(n log n) for sorting plus O(n log n) for the prefix-sums
        and binary searches, which is acceptable for n up to 1000.
        """

        from collections import Counter
        import bisect

        # 1) Count frequencies of all characters
        freq = Counter("".join(words))
        total_pairs = sum(v // 2 for v in freq.values())
        leftover_singles = sum(v % 2 for v in freq.values())

        # 2) Build arrays p0 (even-length words) and p1 (odd-length words),
        #    where each entry is how many pairs that word needs.
        p0 = []  # for words with s_i = 0
        p1 = []  # for words with s_i = 1

        for w in words:
            L = len(w)
            p = L // 2
            s = L % 2
            if s == 0:
                p0.append(p)
            else:
                p1.append(p)

        # Sort them ascending by p_i
        p0.sort()
        p1.sort()

        # Prefix sums for easy pair-sum checks: P0[i] = sum of p0[:i]
        P0 = [0]
        for val in p0:
            P0.append(P0[-1] + val)

        # P1[i] = sum of p1[:i]
        P1 = [0]
        for val in p1:
            P1.append(P1[-1] + val)

        # A helper to pick as many even-length words as possible given leftover pairs
        # We'll do a binary search on P0 to find the max j with P0[j] <= pairs.
        def pickmax0(pairs):
            # We want the largest j s.t. P0[j] <= pairs
            # P0 is sorted ascending, we can do bisect
            # bisect_right returns the insertion point to maintain sorted order.
            # We want the greatest j with P0[j] <= pairs => j = bisect_right(P0, pairs) - 1
            # but j must not exceed len(p0).
            j = bisect.bisect_right(P0, pairs) - 1
            return max(0, j)  # number of words from p0

        # A helper to pick additional odd-length words from p1 starting at index start_idx,
        # given leftover pairs = pairs and leftover singles = singles. We do the same approach
        # of using the prefix sums P1, but we need to offset by P1[start_idx].
        def pickmax1(start_idx, pairs, singles):
            # We can only pick up to singles many from p1 (each requires 1 leftover single).
            # We want the largest u s.t. P1[start_idx + u] - P1[start_idx] <= pairs and u <= singles
            # We'll do binary search on u in [0 .. len(p1) - start_idx].
            lo, hi = 0, min(len(p1) - start_idx, singles)
            base = P1[start_idx]
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if P1[start_idx + mid] - base <= pairs:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        # We'll try all possible x = how many from p1 we pick first (the smallest p1's).
        # Then fill p0, then pick more from p1.
        n0 = len(p0)
        n1 = len(p1)
        ans = 0

        # Precompute the maximum direct pick from p1 if we do not try partial picking first
        # (just as a quick check, though the loop below should cover it).
        # We'll simply pick from p1 in ascending order until we run out of pairs or singles.
        # Then pick from p0 in ascending order with remaining pairs.
        # This is effectively the "pick from p1 first" scenario in full, but we'll do
        # a more systematic version in the loop below anyway.
        # Either way, the loop approach covers this scenario exactly.

        limit = min(n1, leftover_singles)
        for x in range(limit + 1):
            # x = how many of the smallest (p1) words we pick first
            # check if we have enough pairs
            needed_pairs_for_first_x = P1[x]
            if needed_pairs_for_first_x > total_pairs:
                # can't pick x from p1 initially
                break
            # we can pick x words from p1
            leftover_pairs = total_pairs - needed_pairs_for_first_x
            leftover_sing = leftover_singles - x

            # now pick as many as possible from p0
            y = pickmax0(leftover_pairs)
            leftover_pairs2 = leftover_pairs - P0[y]

            # now try picking more from p1 beyond x
            u = pickmax1(x, leftover_pairs2, leftover_sing)
            ans = max(ans, x + y + u)

        return ans