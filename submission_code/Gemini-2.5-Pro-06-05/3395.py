import collections

class Solution:
    def minAnagramLength(self, s: str) -> int:
        """
        Given a string s, which is known to be a concatenation of anagrams of some string t,
        this function returns the minimum possible length of the string t.

        Args:
            s: The input string.

        Returns:
            The minimum possible length of the anagram base string t.
        """
        n = len(s)

        # Iterate through all possible lengths `L` for the base string `t`.
        # The length `L` must be a divisor of the total length `n`.
        for L in range(1, n + 1):
            if n % L == 0:
                # This length `L` is a candidate. We need to verify if `s` can be
                # partitioned into `n/L` segments, all being anagrams of each other.

                # We can take the first segment as the reference. All other segments
                # must be an anagram of this one.
                first_segment_counts = collections.Counter(s[0:L])

                # Assume the length is valid until we find a contradiction.
                is_valid = True
                
                # Iterate through the remaining segments of length `L`.
                for i in range(L, n, L):
                    current_segment = s[i : i + L]
                    current_segment_counts = collections.Counter(current_segment)
                    
                    if current_segment_counts != first_segment_counts:
                        # Found a segment that is not an anagram of the first one.
                        # This length `L` is not valid.
                        is_valid = False
                        break
                
                # If the loop completed without finding any non-anagrammatic segments,
                # then `L` is a valid length.
                if is_valid:
                    # Since we are iterating `L` from 1 upwards, the first valid
                    # length we find is the minimum possible length.
                    return L
        
        # This part is theoretically unreachable due to the problem's constraints.
        # When L=n, the check is trivially true, and n will be returned if no smaller
        # L was found.
        return n