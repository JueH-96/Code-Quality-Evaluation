class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        """
        We want the lexicographically largest substring that can appear as one segment
        when splitting 'word' into 'numFriends' non-empty contiguous parts in all
        possible ways. A key insight is that this is the same as finding the
        lexicographically largest prefix of some suffix of 'word' that is valid
        under the splitting constraints.

        By building the suffix array (SA) of 'word' in ascending lexicographical order,
        the last entry in SA corresponds to the lexicographically largest suffix.
        Any prefix of this largest suffix is guaranteed to be lexicographically
        larger than any substring from smaller suffixes.

        The main remaining detail is to figure out how long that segment (prefix)
        can be while still allowing a valid k-way split. One can derive that:
          - If p < (k - 1), then the segment length L <= min(n - p, n - k + 1).
          - Else, L <= (n - p).

        Where p is the start index of the suffix in the original string, n = len(word).

        We'll therefore:
          1. Construct the suffix array (SA) for 'word'.
          2. Take p = SA[-1] (index of the largest suffix).
          3. Compute the allowed segment length L.
          4. Return word[p : p + L].

        This yields the lexicographically largest possible segment in the "box".
        """

        s = word
        n = len(s)
        k = numFriends

        # Build suffix array in O(n log n) time using a standard doubling approach.
        def build_suffix_array(text):
            """Return the suffix array (SA) of 'text' in ascending lexicographical order."""
            n_ = len(text)
            # Initial ranks from the character codes, plus a sentinel (-1) at the end
            rank_ = [ord(c) for c in text] + [-1]
            sa_ = list(range(n_))  # Suffix array positions
            temp = [0] * (n_ + 1)

            length = 1
            while True:
                # Sort by (rank[i], rank[i+length]) pairs
                array = [(rank_[i], rank_[i + length] if i + length < n_ else -1, i) 
                         for i in range(n_)]
                array.sort()
                for i in range(n_):
                    sa_[i] = array[i][2]
                # Recompute ranks
                temp[sa_[0]] = 0
                for i in range(1, n_):
                    left = array[i - 1]
                    right = array[i]
                    temp[sa_[i]] = temp[sa_[i - 1]]
                    if (right[0] != left[0]) or (right[1] != left[1]):
                        temp[sa_[i]] += 1
                for i in range(n_):
                    rank_[i] = temp[i]

                # If the highest rank is n_-1, all ranks are distinct, we can stop
                if rank_[sa_[-1]] == n_ - 1:
                    break
                length <<= 1
                if length >= n_:
                    break
            return sa_

        sa = build_suffix_array(s)

        # p is the start index of the lexicographically largest suffix
        p = sa[-1]

        # Compute how large a segment we can take from that suffix
        # Case 1: p < (k - 1) => segment length <= min(n - p, n - k + 1)
        # Case 2: p >= (k - 1) => segment length <= (n - p)
        if p < k - 1:
            L = min(n - p, n - k + 1)
        else:
            L = n - p

        return s[p : p + L]