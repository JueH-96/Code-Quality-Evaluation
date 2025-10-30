class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # collect the indices where two strings differ
        diff_idx = [i for i, (a, b) in enumerate(zip(s1, s2)) if a != b]
        m = len(diff_idx)

        # each operation flips exactly two positions, therefore
        # the amount of mismatches must be even
        if m & 1:
            return -1
        if m == 0:
            return 0

        # optimal strategy:
        #   pair the i-th mismatch with the (i+1)-th mismatch (i is even),
        #   pay min(distance, x) for each pair
        # proof sketch:
        #   – using only adjacent-flip operations, removing the two
        #     mismatches at positions p and q (p < q) costs (q-p)
        #   – a “free-choice” flip removes any two mismatches for cost x
        #   – cost function f(d) = min(d, x) is non-decreasing in d
        #     ⇒ for four ordered mismatches a<b<c<d,
        #        f(a,c)+f(b,d) ≥ f(a,b)+f(c,d)
        #     so crossing pairs can never be cheaper than consecutive pairs
        #
        # Hence consecutive pairing is always optimal.
        ans = 0
        for i in range(0, m, 2):
            distance = diff_idx[i + 1] - diff_idx[i]
            ans += min(distance, x)

        return ans