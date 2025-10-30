class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        """
        Finds the smallest starting index of a substring in s that is almost equal to pattern.
        Almost equal means at most one character difference (Hamming distance <= 1).
        This solution uses a polynomial rolling hash with binary search for efficiency.
        """
        n = len(s)
        m = len(pattern)

        if m > n:
            return -1

        # Using two hash functions to minimize collisions
        B1, M1 = 31, 1_000_000_007
        B2, M2 = 37, 1_000_000_009

        # Precompute powers of bases. P1[k] = B1^k, P2[k] = B2^k
        P1 = [1] * (n + 1)
        P2 = [1] * (n + 1)
        for i in range(1, n + 1):
            P1[i] = (P1[i - 1] * B1) % M1
            P2[i] = (P2[i - 1] * B2) % M2

        # Helper to convert char to int > 0, to avoid issues with hash value 0
        def char_to_int(c):
            return ord(c) - ord('a') + 1

        # Precompute prefix hashes for s
        H_s1 = [0] * (n + 1)
        H_s2 = [0] * (n + 1)
        for i in range(n):
            val = char_to_int(s[i])
            H_s1[i + 1] = (H_s1[i] * B1 + val) % M1
            H_s2[i + 1] = (H_s2[i] * B2 + val) % M2

        # Precompute prefix hashes for pattern
        H_p1 = [0] * (m + 1)
        H_p2 = [0] * (m + 1)
        for i in range(m):
            val = char_to_int(pattern[i])
            H_p1[i + 1] = (H_p1[i] * B1 + val) % M1
            H_p2[i + 1] = (H_p2[i] * B2 + val) % M2

        def get_s_hash(i: int, j: int) -> tuple[int, int]:
            # Returns the double hash of substring s[i:j]
            if i >= j:
                return (0, 0)
            length = j - i
            h1 = (H_s1[j] - (H_s1[i] * P1[length]) % M1 + M1) % M1
            h2 = (H_s2[j] - (H_s2[i] * P2[length]) % M2 + M2) % M2
            return (h1, h2)

        def get_p_hash(i: int, j: int) -> tuple[int, int]:
            # Returns the double hash of substring pattern[i:j]
            if i >= j:
                return (0, 0)
            length = j - i
            h1 = (H_p1[j] - (H_p1[i] * P1[length]) % M1 + M1) % M1
            h2 = (H_p2[j] - (H_p2[i] * P2[length]) % M2 + M2) % M2
            return (h1, h2)

        pat_hash = get_p_hash(0, m)

        for i in range(n - m + 1):
            # Check for 0 differences (perfect match)
            if get_s_hash(i, i + m) == pat_hash:
                return i

            # Check for 1 difference
            # Binary search to find the first mismatch position relative to the window
            low, high = 0, m - 1
            mismatch_pos = m
            while low <= high:
                mid = (low + high) // 2
                # Check prefix of length mid + 1
                if get_s_hash(i, i + mid + 1) == get_p_hash(0, mid + 1):
                    low = mid + 1
                else:
                    mismatch_pos = mid
                    high = mid - 1

            if mismatch_pos < m:
                # A mismatch was found at `mismatch_pos`.
                # Check if the suffixes after the mismatch are identical.
                suffix_hash_s = get_s_hash(i + mismatch_pos + 1, i + m)
                suffix_hash_p = get_p_hash(mismatch_pos + 1, m)

                if suffix_hash_s == suffix_hash_p:
                    return i
        
        return -1