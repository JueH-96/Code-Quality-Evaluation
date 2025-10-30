class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        # If we need zero substrings, that's always possible.
        if k == 0:
            return True

        # Compute the first (leftmost) and last (rightmost) occurrence of each character.
        L = [n] * 26
        R = [-1] * 26
        for i, ch in enumerate(s):
            idx = ord(ch) - ord('a')
            if i < L[idx]:
                L[idx] = i
            if i > R[idx]:
                R[idx] = i

        intervals = []
        # For each position i that is the first occurrence of s[i],
        # try to build the minimal "valid" interval starting there.
        for i, ch in enumerate(s):
            idx = ord(ch) - ord('a')
            # only start from the first occurrence of this character
            if L[idx] != i:
                continue

            start = i
            end = R[idx]
            j = start
            valid = True
            # Expand end to cover all characters whose R[] lies within,
            # and check that no character inside has a first occurrence before start.
            while j <= end:
                c2 = ord(s[j]) - ord('a')
                if L[c2] < start:
                    valid = False
                    break
                if R[c2] > end:
                    end = R[c2]
                j += 1

            # If valid and not the entire string, record it.
            if valid and not (start == 0 and end == n - 1):
                intervals.append((start, end))

        # Greedily pick as many non-overlapping intervals as possible (sort by end).
        intervals.sort(key=lambda x: x[1])
        count = 0
        last_end = -1
        for st, ed in intervals:
            if st > last_end:
                count += 1
                last_end = ed
                if count >= k:
                    return True

        return False