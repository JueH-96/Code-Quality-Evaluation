from bisect import bisect_right, bisect_left

def kmp_occurrences(text: str, pattern: str):
    """Returns a list of all starting indices where pattern occurs in text using KMP."""
    if pattern == "":
        # If the pattern is empty, we can think of it as occurring at every index.
        # In our usage, we do not want to list every index because that can be huge.
        # Instead, we will treat empty pattern as a special case.
        return None  
    # Build prefix table
    m = len(pattern)
    lps = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j-1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j
    occurrences = []
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = lps[j-1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            occurrences.append(i - m + 1)
            j = lps[j-1]
    return occurrences

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # p contains exactly two '*' characters.
        # Split p into three parts: A, B, C.
        parts = p.split('*')
        if len(parts) != 3:
            # Just in case, but by problem spec there are exactly two stars.
            return -1
        A, B, C = parts[0], parts[1], parts[2]
        
        # Special-case: if all parts are empty then pattern is all wildcards.
        if A == "" and B == "" and C == "":
            return 0
        
        # Helper: get occurrences using KMP when literal is nonempty.
        occA = kmp_occurrences(s, A) if A != "" else None
        occB = kmp_occurrences(s, B) if B != "" else None
        occC = kmp_occurrences(s, C) if C != "" else None

        INF = float('inf')
        ans = INF
        
        # The matching procedure:
        # We need to pick indices a, b, c in s such that:
        #   a is the start of an occurrence of A (if A nonempty) 
        #   b is the start of an occurrence of B (if B nonempty)
        #   c is the start of an occurrence of C (if C nonempty)
        # with the conditions (in T = s[a : (c + len(C))] if C nonempty or s[a : (b+len(B))] if C is empty):
        #   if A nonempty: a + len(A) <= b   [so that A is matched at the very beginning]
        #   if B nonempty: b + len(B) <= c   [so that B is followed by C]
        # When any literal is empty, we simulate its “matching”
        #
        # We break into two cases: when B is nonempty and when B is empty.
        
        if B != "":
            # In this case we iterate over each occurrence of B.
            # For each occurrence of B at index b, we want to choose:
            #   a: an occurrence of A such that (if A nonempty) a <= b - len(A)
            #      if A is empty, we can set a = b (the empty string matches at any position)
            #   c: an occurrence of C such that (if C nonempty) c >= b + len(B)
            #      if C is empty, we set c = b + len(B).
            for b in occB:
                # choose a from A
                if A != "":
                    # We need an occurrence a such that a <= b - len(A)
                    # Since occurrences are sorted in increasing order, we want the maximum a that does not exceed (b - len(A)).
                    limit = b - len(A)
                    idx = bisect_right(occA, limit) - 1
                    if idx < 0:
                        continue
                    a_start = occA[idx]
                    # Also, ensure that a_start + len(A) <= b. (It must, since a_start <= b - len(A))
                    if a_start + len(A) > b:
                        continue
                else:
                    a_start = b  # if A is empty, we treat the matching start as b.
                
                # choose c from C
                if C != "":
                    threshold = b + len(B)
                    idx_c = bisect_left(occC, threshold)
                    if idx_c == len(occC):
                        continue
                    c_start = occC[idx_c]
                    # We require that c_start >= b + len(B)
                else:
                    # if C is empty, we simulate by letting the match for C end at b+len(B)
                    c_start = b + len(B)
                
                # The candidate matching substring would be T = s[a_start : (c_start + (len(C) if C != "" else 0))].
                total_end = c_start + (len(C) if C != "" else 0)
                cur_len = total_end - a_start
                if cur_len < ans:
                    ans = cur_len
        else:
            # B is empty, so pattern is A + "*" + "" + "*" + C, i.e. A and C must appear (in that order) but with nothing mandatory in the middle.
            # The matching algorithm:
            #   - First part: if A is nonempty, we pick an occurrence a.
            #   - Then, we need to match C after (or possibly overlapping) the match for A.
            #     How do we simulate the sequential matching? In standard wildcard matching,
            #     we match A at the beginning; then since B is empty, the next literal is C which is matched
            #     at some index c, with the requirement that the search for C starts at index (a + len(A)).
            #   - If A is empty, then we only require that C appear (or if C is also empty, then empty match).
            if A != "" and C != "":
                for a in occA:
                    # After matching A at index a, the next literal C must occur at some index c >= a + len(A).
                    threshold = a + len(A)
                    idx_c = bisect_left(occC, threshold)
                    if idx_c == len(occC):
                        continue
                    c_start = occC[idx_c]
                    total_end = c_start + len(C)
                    cur_len = total_end - a
                    if cur_len < ans:
                        ans = cur_len
            elif A == "" and C != "":
                # Pattern is "" + "**" + C, so any occurrence of C produces a match.
                # Minimal window is just the occurrence of C, i.e. T = s[c : c+len(C)]
                # So answer is len(C) if C occurs.
                if occC:
                    ans = min(ans, len(C))
            elif A != "" and C == "":
                # Pattern is A + "**" + "", so any occurrence of A produces a match.
                # Minimal window is just the A-part.
                if occA:
                    ans = min(ans, len(A))
            else:
                # Both A and C are empty (and also B is empty, but that was covered above).
                ans = 0

        if ans == INF:
            return -1
        return ans