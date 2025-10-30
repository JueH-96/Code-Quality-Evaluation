class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict

        # Collect the lengths of contiguous runs for each character
        runs = defaultdict(list)
        n = len(s)
        i = 0
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            runs[s[i]].append(j - i)
            i = j

        ans = -1
        # For each character, check possible substring lengths k
        for c, lengths in runs.items():
            max_run = max(lengths)
            # Try all k from 1 up to the longest run of c
            for k in range(1, max_run + 1):
                occ = 0
                # Count how many substrings of length k can be formed in total
                for r in lengths:
                    if r >= k:
                        occ += (r - k + 1)
                        if occ >= 3:
                            # No need to continue counting once we reach 3
                            break
                if occ >= 3:
                    ans = max(ans, k)

        return ans