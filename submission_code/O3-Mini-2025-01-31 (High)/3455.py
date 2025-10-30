class Solution:
    def minimumLength(self, s: str) -> int:
        # Explanation:
        # Every allowed operation works only on the occurrences of one letter.
        # Suppose a letter x occurs f times in s.
        # List its positions in order. An allowed operation requires that we choose one
        # occurrence that is “internal” (i.e. not the first or last in the list of positions)
        # and then remove the closest occurrence to its left and to its right.
        # Notice that an operation on letter x removes 2 copies and leaves the chosen (pivot) copy.
        #
        # Thus, if a letter appears f times:
        #   • If f == 1, there is no internal occurrence so we cannot remove any – we end up with 1 copy.
        #   • If f == 2, again neither occurrence has both a left and right copy so we must keep 2 copies.
        #   • If f >= 3: then every allowed operation removes 2 occurrences.
        #     It turns out that if f is odd you can eventually remove pairs until just one copy remains,
        #     while if f is even you can only reduce it to 2 copies.
        #
        # Since removals on different letters act independently, the minimum possible length
        # is the sum over each distinct letter of:
        #     1 if the frequency is odd (or if it is 1), and 2 if the frequency is even (and at least 2).
        #
        # For example:
        #   s = "abaacbcbb"
        #   Frequency:
        #       a: 3 -> (odd)  becomes 1 copy in best-case.
        #       b: 4 -> (even) becomes 2 copies.
        #       c: 2 -> remains 2 copies.
        #   So answer = 1 + 2 + 2 = 5.
        #
        # The solution below computes a frequency counter and then sums the minimal leftover for
        # each letter.
        
        from collections import Counter
        freq = Counter(s)
        ans = 0
        for count in freq.values():
            if count < 3:
                ans += count
            else:
                # For count >= 3,
                # odd count can be reduced to 1 ("all but one removed")
                # even count can be reduced to 2.
                ans += 1 if count % 2 == 1 else 2
        return ans