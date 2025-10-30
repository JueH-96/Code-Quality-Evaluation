class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        # k special substrings are asked for; if k==0, we always can choose 0 of them.
        n = len(s)
        if k == 0:
            return True
        
        # Precompute the first and last occurrence for each character.
        first = {}
        last = {}
        for i, ch in enumerate(s):
            if ch not in first:
                first[ch] = i
            last[ch] = i
        
        # Key observation:
        # A substring s[l:r+1] is "special" if for every character c in it,
        # all the occurrences of c in s lie entirely in [l, r]. In particular,
        # for c = s[l] we must have first[c] == l.
        #
        # So it is natural to restrict our attention to substrings that start at an index l
        # where l is the first occurrence of s[l]. Then we “grow” the substring as needed.
        
        candidates = []
        # Only consider l if s[l] is appearing the first time in s.
        for l in range(n):
            if first[s[l]] != l:
                continue
            # The minimal candidate must at least include all occurrences of s[l].
            r = last[s[l]]
            valid = True
            i = l
            # Expand the candidate to include the full range for every character encountered.
            while i <= r:
                # For any letter in the substring, if its very first occurrence is before l, then
                # it means the letter appears outside s[l:r+1] and so s[l:r+1] violates the special condition.
                if first[s[i]] < l:
                    valid = False
                    break
                # Ensure that the candidate covers all occurrences of s[i].
                if last[s[i]] > r:
                    r = last[s[i]]
                i += 1
            # Also, the special substring must not be the entire string.
            if valid and not (l == 0 and r == n - 1):
                candidates.append((l, r))
        
        # Now, we want to see if we can choose at least k disjoint candidate intervals.
        # We can do this by the standard greedy selection: choose intervals (by sorted ending position)
        # and count how many do not overlap.
        candidates.sort(key=lambda interval: interval[1])
        count = 0
        end_last = -1
        for l, r in candidates:
            if l > end_last:
                count += 1
                end_last = r
        return count >= k

# Testing the solution with the provided examples and some additional cases.
if __name__ == '__main__':
    sol = Solution()
    
    # Example 1:
    s1 = "abcdbaefab"
    k1 = 2
    print(sol.maxSubstringLength(s1, k1))  # Expected: True
    
    # Example 2:
    s2 = "cdefdc"
    k2 = 3
    print(sol.maxSubstringLength(s2, k2))  # Expected: False
    
    # Example 3:
    s3 = "abeabe"
    k3 = 0
    print(sol.maxSubstringLength(s3, k3))  # Expected: True

    # Additional test:
    # In "abba", one valid special substring is "bb" (from index 1 to 2).
    s4 = "abba"
    k4 = 1
    print(sol.maxSubstringLength(s4, k4))  # Expected: True

    # In "aaa", any substring covering 'a' would either be a part
    # of the full string or miss some occurrence of 'a'.
    s5 = "aaa"
    k5 = 1
    print(sol.maxSubstringLength(s5, k5))  # Expected: False