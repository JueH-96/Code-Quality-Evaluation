from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        """
        Finds beautiful indices in string s.

        An index i is beautiful if:
        1. s[i..(i + a.length - 1)] == a
        2. There exists an index j such that s[j..(j + b.length - 1)] == b and |j - i| <= k.

        Args:
            s: The input string.
            a: The first pattern string.
            b: The second pattern string.
            k: The maximum allowed difference between indices i and j.

        Returns:
            A sorted list of beautiful indices.
        """

        def find_all_occurrences(text: str, pattern: str) -> List[int]:
            """Finds all starting indices of pattern in text."""
            indices = []
            start = 0
            while True:
                index = text.find(pattern, start)
                if index == -1:
                    break
                indices.append(index)
                start = index + 1 # Move start past the found index to find next occurrence
            return indices

        # 1. Find all occurrences of a and b
        a_indices = find_all_occurrences(s, a)
        b_indices = find_all_occurrences(s, b)

        # If 'a' is not found, no index can be beautiful
        if not a_indices:
            return []
        
        # If 'b' is not found, no index can be beautiful
        if not b_indices:
            return []

        # 2. Find beautiful indices using a two-pointer approach
        # Since a_indices and b_indices are already sorted from find_all_occurrences,
        # we can use two pointers.
        beautiful_indices = []
        ptr_b = 0 # Pointer for b_indices

        for i in a_indices:
            # For the current index i from a_indices, we need to find if there's
            # any j in b_indices such that i - k <= j <= i + k.

            # Advance ptr_b to the first index in b_indices that is >= i - k.
            # We don't need to check b_indices before ptr_b because i is increasing,
            # so any previous j (b_indices < current ptr_b) that was less than i - k
            # will also be less than the next i' - k (where i' >= i).
            while ptr_b < len(b_indices) and b_indices[ptr_b] < i - k:
                ptr_b += 1

            # Now, b_indices[ptr_b] is the first potential j >= i - k.
            # Check if this potential j is also <= i + k.
            # If it is, then we found a valid j for the current i, so i is beautiful.
            # We only need to find *one* such j, so the first one >= i-k is sufficient
            # to check if it's also <= i+k. If it is, i is beautiful.
            if ptr_b < len(b_indices) and b_indices[ptr_b] <= i + k:
                beautiful_indices.append(i)
            
            # Note: We do NOT reset ptr_b for the next i. It continues from
            # where it left off, which is key for the O(len(a_indices) + len(b_indices)) complexity.
            # If the current b_indices[ptr_b] was > i + k, it means no j at or after ptr_b
            # could satisfy j <= i + k. However, for the *next* i in a_indices (which is larger),
            # the range [i' - k, i' + k] shifts to the right, so b_indices[ptr_b] might
            # fall into that range. That's why we don't increment ptr_b if the current
            # b_indices[ptr_b] is > i + k; it might be the first valid candidate for the next i.

        # The beautiful_indices list is automatically sorted because we iterate
        # through a_indices in sorted order.
        return beautiful_indices