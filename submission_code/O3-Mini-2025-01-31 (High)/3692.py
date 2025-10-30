import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # The pattern p contains exactly two '*' characters.
        # We split p into three parts: p1, p2, p3.
        # A substring T of s matches p if and only if
        #   T = p1 + X + p2 + Y + p3   (for some (possibly empty) strings X and Y),
        # with the requirement that:
        #   • T starts with p1,
        #   • p2 occurs (non overlapping) after p1, and
        #   • T ends with p3.
        #
        # Our strategy is to precompute the occurrences (starting indices) in s for each fixed segment.
        # Then we iterate over possible start indices x (where p1 occurs at position x)
        # and use binary search into the occurrences lists for p2 and p3 (adjusting by lengths)
        # to “chain” a valid substring cover.
        
        # Split p into its three parts (exactly 2 stars will result in 3 parts)
        parts = p.split('*')
        p1, p2, p3 = parts[0], parts[1], parts[2]
        n = len(s)
        
        # Special-case: If all parts are empty, the empty substring is valid.
        if p1 == "" and p2 == "" and p3 == "":
            return 0

        # Helper function: KMP search to find all occurrences of pattern pat in text.
        # If pat is empty, we return every index (from 0 to len(text)) as a valid match.
        def kmp_search(text, pat):
            if pat == "":
                return list(range(len(text) + 1))
            # Build lps (Longest Prefix Suffix) array for pat:
            lps = [0] * len(pat)
            j = 0
            for i in range(1, len(pat)):
                while j > 0 and pat[i] != pat[j]:
                    j = lps[j - 1]
                if pat[i] == pat[j]:
                    j += 1
                    lps[i] = j

            res = []
            j = 0
            for i in range(len(text)):
                while j > 0 and text[i] != pat[j]:
                    j = lps[j - 1]
                if text[i] == pat[j]:
                    j += 1
                    if j == len(pat):
                        res.append(i - j + 1)
                        j = lps[j - 1]
            return res

        # Precompute occurrences for each fixed part.
        # For nonempty patterns, we use KMP.
        # For an empty fixed part, any index (from 0 to n, or 0 to n-1 for p1)
        # qualifies. For the matching purpose here, we want to “simulate” a match
        # of the empty pattern at any possible boundary.
        if p1 != "":
            L1 = kmp_search(s, p1)
        else:
            # For matching T, the starting index can be any index from 0 to n.
            L1 = list(range(n + 1))
        if p2 != "":
            L2 = kmp_search(s, p2)
        else:
            L2 = list(range(n + 1))
        if p3 != "":
            L3 = kmp_search(s, p3)
        else:
            L3 = list(range(n + 1))
        
        # L1, L2, L3 are in sorted order.
        # For a substring T = s[x:cand_end] to match the pattern,
        # we require:
        #   - T must start with p1, so x must be an index in L1.
        #   - Immediately after p1, that is at index (x + len(p1)), we need to match p2.
        #     If p2 is nonempty, let candidate_y be the earliest occurrence in L2 
        #     that is at or after x + len(p1). If p2 is empty, we can set candidate_y = x+len(p1).
        #   - The segment p2 (if nonempty) occupies candidate_y to candidate_y+len(p2).
        #     Let required_z = candidate_y + (len(p2) if p2 else 0).
        #     Then if p3 is nonempty, we need to find an occurrence in L3 with starting index >= required_z;
        #     otherwise if p3 is empty, we set candidate end = required_z.
        #   - Finally, if p3 is nonempty, the substring T must end exactly at candidate_z+len(p3).
        # The length of T is then (candidate_end - x). We wish to minimize that.
        ans = float('inf')
        
        for x in L1:
            # For a valid match, T must start at index x;
            # the fixed prefix p1 is matched in s[x:x+len(p1)].
            y_target = x + len(p1)  # earliest index where p2 can start
            # Find p2 occurrence:
            if p2 != "":
                pos = bisect.bisect_left(L2, y_target)
                if pos == len(L2):
                    continue  # no occurrence of p2 after index y_target
                candidate_y = L2[pos]
            else:
                candidate_y = y_target  # empty pattern matches at y_target

            # After p2 is matched, if it is nonempty, it occupies positions candidate_y : candidate_y+len(p2).
            if p2 != "":
                required_z = candidate_y + len(p2)
            else:
                required_z = candidate_y  # if p2 is empty, no additional length

            # Next, find p3 occurrence:
            if p3 != "":
                pos2 = bisect.bisect_left(L3, required_z)
                if pos2 == len(L3):
                    continue  # no occurrence of p3 that starts at or after required_z
                candidate_z = L3[pos2]
                candidate_end = candidate_z + len(p3)
            else:
                candidate_end = required_z  # if p3 is empty, T ends at required_z
            
            # The candidate substring T is s[x:candidate_end]; check that it is within s.
            if candidate_end > n:
                continue

            candidate_length = candidate_end - x
            ans = min(ans, candidate_length)
        
        return ans if ans != float('inf') else -1