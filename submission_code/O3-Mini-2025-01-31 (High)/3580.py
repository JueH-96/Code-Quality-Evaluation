class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        mod = 10**9 + 7
        base = 131

        # Precompute powers up to max(n, m)
        size = max(n, m)
        power = [1] * (size + 1)
        for i in range(1, size + 1):
            power[i] = (power[i - 1] * base) % mod

        # Precompute prefix hashes for s and pattern.
        # We map 'a'->1, 'b'->2, ... by using ord(ch) - 96.
        Hs = [0] * (n + 1)
        Hp = [0] * (m + 1)
        for i in range(n):
            Hs[i + 1] = (Hs[i] * base + (ord(s[i]) - 96)) % mod
        for i in range(m):
            Hp[i + 1] = (Hp[i] * base + (ord(pattern[i]) - 96)) % mod

        # Helper function: returns hash for substring [l, r) from a prefix hash array H
        def get_hash(H, l, r):
            return (H[r] - H[l] * power[r - l]) % mod

        # For every candidate starting index i we want to check whether
        # the substring s[i:i+m] is almost equal to pattern:
        # "almost equal" means that by changing at most one character in s[i:i+m],
        # we can make it identical to pattern.
        #
        # One easy case: if the substring exactly equals pattern, return i.
        # Otherwise, if there is exactly one mismatch, then we want:
        #   Let L be the number of matching characters from the beginning.
        #   Then s[i+L] != pattern[L] (the first mismatch) and s[i+L+1:i+m] exactly equals pattern[L+1:m].
        #
        # We use binary search (with rolling hash equality checks) to compute the max L such that
        # s[i:i+L] == pattern[0:L].

        for i in range(n - m + 1):
            # Check for an exact match.
            if get_hash(Hs, i, i + m) == get_hash(Hp, 0, m):
                return i

            # Binary search to find the longest matching prefix length L.
            lo, hi = 0, m
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if get_hash(Hs, i, i + mid) == get_hash(Hp, 0, mid):
                    lo = mid
                else:
                    hi = mid - 1
            L = lo
            # If L < m then s[i+L] is the first position of discrepancy.
            # For the substring to be fixable with one change,
            # the remainder must match exactly.
            if L < m:
                if get_hash(Hs, i + L + 1, i + m) == get_hash(Hp, L + 1, m):
                    return i
        return -1


# ======= Testing =======
if __name__ == '__main__':
    sol = Solution()

    # Example 1:
    s1 = "abcdefg"
    pattern1 = "bcdffg"
    print(sol.minStartingIndex(s1, pattern1))  # Expected output: 1

    # Example 2:
    s2 = "ababbababa"
    pattern2 = "bacaba"
    print(sol.minStartingIndex(s2, pattern2))  # Expected output: 4

    # Example 3:
    s3 = "abcd"
    pattern3 = "dba"
    print(sol.minStartingIndex(s3, pattern3))  # Expected output: -1

    # Example 4:
    s4 = "dde"
    pattern4 = "d"
    print(sol.minStartingIndex(s4, pattern4))  # Expected output: 0