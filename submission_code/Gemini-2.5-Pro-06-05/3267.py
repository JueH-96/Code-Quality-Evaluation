import collections

class Solution:
    def maximumLength(self, s: str) -> int:
        """
        Finds the length of the longest special substring that occurs at least thrice.
        """
        
        # groups will store a list of lengths for each character's contiguous blocks.
        # e.g., for s = "aaabaaa", groups['a'] will be [3, 3]
        groups = collections.defaultdict(list)
        n = len(s)
        i = 0
        while i < n:
            j = i
            while j < n and s[j] == s[i]:
                j += 1
            length = j - i
            groups[s[i]].append(length)
            i = j
            
        max_len = 0
        
        # Iterate over each character that appears in s
        for char in groups:
            lengths = groups[char]
            # Sort in descending order to easily access the three largest groups
            lengths.sort(reverse=True)
            
            # We pad with 0s to handle cases with fewer than 3 groups of this character.
            # k1, k2, k3 are the lengths of the three longest groups.
            k1 = lengths[0] if len(lengths) > 0 else 0
            k2 = lengths[1] if len(lengths) > 1 else 0
            k3 = lengths[2] if len(lengths) > 2 else 0

            # There are three main ways to get at least 3 identical special substrings.
            # We find the max possible length from each case and take the overall maximum.
            
            # Case 1: All 3+ from the single largest group.
            # Max L = k1 - 2
            max_len = max(max_len, k1 - 2)
            
            # Case 2: 2 from the first group and 1 from the second (or other combinations from two groups).
            # Max L = min(k1 - 1, k2)
            max_len = max(max_len, min(k1 - 1, k2))

            # Case 3: 1 from each of the three largest groups.
            # Max L = k3
            max_len = max(max_len, k3)
        
        return max_len if max_len > 0 else -1