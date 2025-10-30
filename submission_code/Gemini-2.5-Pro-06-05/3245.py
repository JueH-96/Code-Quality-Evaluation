from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        """
        Finds all beautiful indices in a string `s`.

        An index `i` is beautiful if:
        1. s[i:i+len(a)] == a
        2. There exists an index `j` such that s[j:j+len(b)] == b and |j - i| <= k.
        """

        # Helper function to find all starting indices of a pattern in a text.
        # Python's `find` is highly optimized (often using algorithms like Boyer-Moore),
        # making this approach very efficient in practice.
        def find_all(text: str, pattern: str) -> List[int]:
            indices = []
            start = 0
            while True:
                # str.find(sub, start) finds the first occurrence at or after index start.
                idx = text.find(pattern, start)
                if idx == -1:
                    break
                indices.append(idx)
                # Move start to the position after the found occurrence to find the next one.
                start = idx + 1
            return indices

        # Step 1: Find all occurrences of substrings `a` and `b`.
        # The returned lists of indices are naturally sorted.
        a_indices = find_all(s, a)
        b_indices = find_all(s, b)

        # If either pattern doesn't appear, no beautiful indices are possible.
        if not a_indices or not b_indices:
            return []

        # Step 2: Use a two-pointer approach to find beautiful indices.
        # This is more efficient than a nested loop or binary searching for each index.
        # The complexity of this part is O(|a_indices| + |b_indices|).
        beautiful_indices = []
        ptr_b = 0  # Pointer for b_indices

        for i in a_indices:
            # We are looking for a `j` in `b_indices` such that |j - i| <= k,
            # which is equivalent to the condition: i - k <= j <= i + k.

            # Advance ptr_b to the first position where b_indices[ptr_b]
            # is a potential candidate for the current `i`.
            # A candidate `j` must be at least `i - k`. We can safely skip any `j`s
            # that are smaller, as they will also be too small for any subsequent `i`s.
            while ptr_b < len(b_indices) and b_indices[ptr_b] < i - k:
                ptr_b += 1

            # If we've exhausted all of `b`'s indices, we can stop.
            # Any subsequent `i` will have an even larger `i - k`, so no
            # `j` will ever be found.
            if ptr_b == len(b_indices):
                break

            # Now, b_indices[ptr_b] is the smallest j that is >= i - k.
            # We just need to check if this j is also <= i + k to satisfy the full condition.
            if b_indices[ptr_b] <= i + k:
                # We found a valid `j` for the current `i`. So `i` is beautiful.
                beautiful_indices.append(i)
        
        return beautiful_indices