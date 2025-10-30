class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        if k == 0:
            return True
        
        # Precompute first and last occurrence for each letter.
        first = {}
        last = {}
        for i, ch in enumerate(s):
            if ch not in first:
                first[ch] = i
            last[ch] = i
        
        # Use a set to avoid duplicate intervals.
        intervals = []
        # Try each letter that appears in s.
        for ch in first:
            start = first[ch]
            end = last[ch]
            # Expand the interval to cover all letters that occur inside.
            j = start
            valid = True
            while j <= end:
                # The letter at j must have all occurrences inside [start, end]
                # (We need to expand end if the letter appears later in s.)
                # Note: if the letter starts before our chosen start, then our start wasn't minimal,
                # but because we always choose start = first[ch] for some letter ch,
                # for any letter inside [start, end] first_occurrence is >= start.
                if first[s[j]] < start:
                    valid = False
                    break
                end = max(end, last[s[j]])
                j += 1
            # Do not use the entire string.
            if valid and not (start == 0 and end == n - 1):
                intervals.append((start, end))
                
        # Remove duplicate intervals if any, then sort by end.
        intervals = list(set(intervals))
        intervals.sort(key=lambda x: x[1])
        
        # Greedy interval scheduling: select disjoint intervals.
        count = 0
        prev_end = -1
        for (i, j) in intervals:
            if i > prev_end:
                count += 1
                prev_end = j
                if count >= k:
                    return True
        return False


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.maxSubstringLength("abcdbaefab", 2))  # Expected output: True (we can choose "cd" and "ef")
    # Example 2:
    print(sol.maxSubstringLength("cdefdc", 3))      # Expected output: False
    # Example 3:
    print(sol.maxSubstringLength("abeabe", 0))      # Expected output: True