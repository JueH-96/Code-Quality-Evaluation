class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        # If k == 0, we can always pick zero substrings
        if k == 0:
            return True
        
        # Record first and last occurrence of each character
        first = [-1] * 26
        last = [-1] * 26
        for i, ch in enumerate(s):
            idx = ord(ch) - ord('a')
            if first[idx] == -1:
                first[idx] = i
            last[idx] = i
        
        intervals = []
        # Find all minimal special substrings
        for i in range(n):
            ci = ord(s[i]) - ord('a')
            # Only start at the first occurrence of s[i]
            if first[ci] != i:
                continue
            end = last[ci]
            j = i
            valid = True
            # Expand the interval to cover all occurrences of any included char
            while j <= end:
                cj = ord(s[j]) - ord('a')
                # If this char appears before i, we can't make a valid substring here
                if first[cj] < i:
                    valid = False
                    break
                # Extend the end if needed
                if last[cj] > end:
                    end = last[cj]
                j += 1
            if valid:
                # Skip the interval if it is the entire string
                if not (i == 0 and end == n - 1):
                    intervals.append((i, end))
        
        # Greedily select the maximum number of non-overlapping intervals
        intervals.sort(key=lambda x: x[1])
        count = 0
        prev_end = -1
        for l, r in intervals:
            if l > prev_end:
                count += 1
                prev_end = r
                if count >= k:
                    return True
        
        return False