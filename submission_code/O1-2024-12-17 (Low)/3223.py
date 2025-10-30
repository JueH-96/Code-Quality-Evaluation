class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        """
        We say a substring s of 'word' is 'complete' if:
          1) Each character in s occurs exactly k times.
          2) For every pair of adjacent characters c1, c2 in s, 
             the absolute difference in their alphabet positions is <= 2.
        
        Return how many such substrings occur in 'word'.
        
        Key observations to allow a solution that works for length up to 1e5:
         - A 'complete' substring of length L = m*k can contain at most m distinct
           letters (each appearing k times).
         - Because adjacent characters differ by at most 2, the set of distinct letters
           in such a substring cannot be large. In particular, if a substring has m
           distinct letters c1 < c2 < ... < c_m (in alphabetical order), each pair
           is at distance <= 2, so c_m - c1 <= 2*(m-1). Practically, m cannot be
           very large compared to the full alphabet. In fact, m <= 13 if you consider
           letters 'a'..'z' (since 25 / 2 ≈ 12.5).
         - Therefore, for each end index, we only need to consider substrings of length
           up to 13*k that end there. This caps the per-index checks at 13.
         - We can check the adjacency-difference condition in O(1) by preprocessing
           a prefix array of "bad adjacency" bits, and likewise check the exact k-frequency
           condition in O(26) = O(1), by using prefix-frequency arrays.
        
        Overall approach:
         1) Precompute an array badAdj[i] = 1 if |word[i] - word[i-1]| > 2 else 0, for i=1..n-1.
         2) Build a prefix sum array badPrefix, where badPrefix[i] = sum of badAdj up to index i-1.
            Then the substring word[L..R] is valid w.r.t adjacency iff badPrefix[R] - badPrefix[L] == 0.
         3) Build prefix-frequency arrays freq[c][i+1], for c in 0..25 and i in 0..n, where
               freq[c][i+1] = freq[c][i] + (1 if word[i] == chr(c+'a') else 0).
            Then the count of letter c in word[L..R] is freq[c][R+1] - freq[c][L].
         4) For each end in [0..n-1], let possible lengths be ℓ = m*k with m=1..13 (stop if ℓ > end+1).
            Then start = end - ℓ + 1.
            - Check adjacency via badPrefix. If not valid, continue.
            - Check frequencies (in O(26)): count how many letters have exactly k occurrences,
              and confirm that all other letters have 0 occurrences. If x letters have k occurrences,
              then x*k == ℓ => valid => increment answer.
        
        This yields a worst-case of about 13 * 26 * n ~ 338 * n. For n=1e5 this can be
        borderline but is typically feasible in optimized Python or C++ with fast IO.
        We implement carefully in Python.
        """
        n = len(word)
        if n == 0:
            return 0
        
        # 1) Build badAdj array
        badAdj = [0]*(n-1)
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                badAdj[i-1] = 1
        
        # 2) badPrefix: prefix sum of badAdj
        #    badPrefix[i] = number of "bad" adjacency up to (but not including) index i.
        #    So substring word[L..R] is adjacency-valid iff badPrefix[R] - badPrefix[L] == 0.
        badPrefix = [0]*(n+1)
        for i in range(n-1):
            badPrefix[i+1] = badPrefix[i] + badAdj[i]
        badPrefix[n] = badPrefix[n-1]  # fill final
        
        # 3) prefix-frequency arrays freq[c][i], c = 26 letters, i = 0..n
        #    freq[c][i] = how many times letter c appeared up to (but not including) index i in word
        freq = [[0]*(n+1) for _ in range(26)]
        for i, ch in enumerate(word):
            cidx = ord(ch) - ord('a')
            for c in range(26):
                freq[c][i+1] = freq[c][i]
            freq[cidx][i+1] += 1
        
        ans = 0
        
        # 4) For each ending index end, try sub-substrings of length m*k for m=1..13
        #    (or while m*k <= end+1).
        max_m = 13  # at most 13 distinct letters if adjacent-diff <=2
        for end_idx in range(n):
            # largest possible length from (end_idx - length + 1) >= 0
            # length = m*k <= (end_idx+1)
            # => m <= (end_idx+1)//k
            limit_m = min(max_m, (end_idx+1)//k)
            for m in range(1, limit_m+1):
                length = m*k
                start = end_idx - length + 1
                # Check adjacency
                if badPrefix[end_idx] - badPrefix[start] != 0:
                    # adjacency break -> no need to check freq
                    continue
                # Now check freq distribution in O(26)
                count_k = 0
                total_chars = 0
                good = True
                for c in range(26):
                    ccount = freq[c][end_idx+1] - freq[c][start]
                    if ccount == k:
                        count_k += 1
                        total_chars += ccount
                    elif ccount != 0:
                        # if ccount is nonzero but not k, it's invalid
                        good = False
                        break
                if good and total_chars == length:
                    ans += 1
        
        return ans