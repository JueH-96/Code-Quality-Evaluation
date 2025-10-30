class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        # If k is 0, we don't need to select any substring.
        if k == 0:
            return True

        n = len(s)
        # There are 26 lowercase letters.
        first_occ = [-1] * 26
        last_occ = [-1] * 26
        
        # Record the first and last occurrence for each character.
        for i, ch in enumerate(s):
            idx = ord(ch) - ord('a')
            if first_occ[idx] == -1:
                first_occ[idx] = i
            last_occ[idx] = i

        # List to hold candidate intervals [start, end]
        candidates = []
        
        # To ensure a substring T = s[i:j+1] is "special":
        # For every character x in T, all occurrences of x in s must be inside [i,j].
        # Notice that if T is special then for the first character in T,
        # we must have i == first_occ[s[i]]. So we only consider such starting positions.
        i = 0
        while i < n:
            idx = ord(s[i]) - ord('a')
            if i != first_occ[idx]:
                i += 1
                continue

            # Compute the minimal interval [i, j] that must include all occurrences
            # of every letter encountered in this interval.
            j = last_occ[idx]
            r = i
            valid = True
            while r <= j:
                c = s[r]
                c_idx = ord(c) - ord('a')
                # If the first occurrence of this letter is before i,
                # then s[i:j+1] would not contain all occurrences of c.
                if first_occ[c_idx] < i:
                    valid = False
                    break
                # Extend the interval j if needed.
                if last_occ[c_idx] > j:
                    j = last_occ[c_idx]
                r += 1
            if valid:
                # The problem says the special substring must not be the entire string.
                if not (i == 0 and j == n - 1):
                    candidates.append((i, j))
            i += 1
        
        # Now choose a maximum set of disjoint candidate intervals.
        # We'll sort by the end index (greedy selection works for interval scheduling).
        candidates.sort(key=lambda interval: interval[1])
        
        count = 0
        last_end = -1
        for start, end in candidates:
            # If the candidate doesn't overlap with the previously chosen one,
            # then we select it.
            if start > last_end:
                count += 1
                last_end = end
                # Early exit: if we already have at least k disjoint substrings.
                if count >= k:
                    return True
        
        return count >= k