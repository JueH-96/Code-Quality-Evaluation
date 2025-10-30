class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        from bisect import bisect_left, bisect_right

        # Quick shortcut when pattern is "**" (both parts empty)
        if p == "**":
            return 0

        # Split pattern p into three parts (A, B, C)
        # p contains exactly two '*' characters.
        parts = p.split('*')
        # There will be three parts: A, B, C.
        A, B, C = parts[0], parts[1], parts[2]
        
        n = len(s)

        # Helper: find all occurrences of a literal pattern pat in s.
        # Returns a sorted list of indices where pat occurs (s[i:i+len(pat)] == pat).
        def findAllOccurrences(s, pat):
            positions = []
            if pat == "":  # empty string matches at every index
                # We do not need all indices; it will be handled specially,
                # so return an empty list (we use conditions below).
                return positions  
            start = 0
            while True:
                pos = s.find(pat, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            return positions

        A_list = findAllOccurrences(s, A)
        B_list = findAllOccurrences(s, B)
        C_list = findAllOccurrences(s, C)

        # If B or C is non-empty and we did not find any occurrence, matching is impossible.
        if B != "" and not B_list:
            return -1
        if C != "" and not C_list:
            return -1

        # For the case A empty: we will allow any candidate start to be chosen. 
        # In our merging strategy, we treat it as: best a = b.
        # For the case C empty: candidate end is taken as b + len(B) - 1.
        
        ans = float('inf')
        # We now iterate over all occurrences of B.
        for b in B_list:
            # Determine candidate start a.
            if A == "":
                a = b  # we pretend the substring candidate starts at b.
            else:
                # We require an occurrence of A that occurs at an index a where a <= b - len(A)
                # (so that A appears at the beginning of the candidate substring).
                limit = b - len(A)
                # Find rightmost index in A_list that is <= limit.
                idx = bisect_right(A_list, limit) - 1
                if idx < 0:
                    continue  # no valid occurrence of A for this b
                a = A_list[idx]
            # Now candidate substring must have:
            #  - B occurring at index b (with length len(B)) and
            #  - Then C must occur after B: i.e. an occurrence of C starting at index >= b + len(B).
            if C == "":
                # If C is empty, we set candidate end as the end of B.
                candidate_end = b + len(B) - 1
            else:
                lower_bound = b + len(B)
                j = bisect_left(C_list, lower_bound)
                if j == len(C_list):
                    continue  # no occurrence of C after B
                c_start = C_list[j]
                candidate_end = c_start + len(C) - 1
            # Ensure the ordering integrity:
            # Our candidate substring (from index a to candidate_end) will look like:
            #  [A][... maybe stuff...][B][... maybe stuff...][C]
            # We already ensured: a+len(A) <= b and b+len(B) <= candidate_end - (len(C)-1).
            # Compute candidate length.
            candidate_length = candidate_end - a + 1
            if candidate_length < ans:
                ans = candidate_length

        # If ans remains infinity, then we didn't get any match.
        if ans == float('inf'):
            return -1
        return ans


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    s1 = "abaacbaecebce"
    p1 = "ba*c*ce"
    print(sol.shortestMatchingSubstring(s1, p1))  # Expected output: 8
    
    # Example 2:
    s2 = "baccbaadbc"
    p2 = "cc*baa*adb"
    print(sol.shortestMatchingSubstring(s2, p2))  # Expected output: -1
    
    # Example 3:
    s3 = "a"
    p3 = "**"
    print(sol.shortestMatchingSubstring(s3, p3))  # Expected output: 0
    
    # Example 4:
    s4 = "madlogic"
    p4 = "*adlogi*"
    print(sol.shortestMatchingSubstring(s4, p4))  # Expected output: 6