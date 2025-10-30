import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        def find_all_occurrences(text, sub):
            if not sub:
                # Empty string matches at every position (before char 0, between 0-1, ..., after last char N)
                # Indices are 0, 1, ..., len(text)
                return list(range(len(text) + 1))
            occurrences = []
            start = -1
            while True:
                start = text.find(sub, start + 1)
                if start == -1:
                    break
                occurrences.append(start)
            return occurrences

        idx1 = p.find('*')
        idx2 = p.find('*', idx1 + 1) # Find second '*' after the first one

        p1 = p[:idx1]
        p2 = p[idx1 + 1 : idx2]
        p3 = p[idx2 + 1 :]

        len_p1 = len(p1)
        len_p2 = len(p2)
        len_p3 = len(p3)

        # Minimum possible length of the substring (when '*' match empty strings)
        min_core_len = len_p1 + len_p2 + len_p3

        # If the minimum required length of the core parts is greater than s length, impossible
        if min_core_len > len(s):
            return -1

        # Find all occurrences of the parts in s
        starts_p1 = find_all_occurrences(s, p1)
        starts_p2 = find_all_occurrences(s, p2)
        starts_p3 = find_all_occurrences(s, p3)
        
        # If any non-empty part doesn't occur in s, no match possible
        # Note: find_all_occurrences returns [] for non-empty sub not found.
        # It returns [0, ..., len(s)] for empty sub.
        if len_p1 > 0 and not starts_p1: return -1
        if len_p2 > 0 and not starts_p2: return -1
        if len_p3 > 0 and not starts_p3: return -1


        min_len = float('inf')

        # Iterate through possible ending positions j of the substring s[i:j]
        # The substring is s[i : j], length is j-i. j is exclusive end index.
        # The minimum possible value for j is min_core_len, when the substring is exactly the concatenation of p1, p2, p3.
        # The maximum possible value for j is len(s).
        for j in range(min_core_len, len(s) + 1):
            # The suffix p3 must match s[j - len_p3 : j]
            end_p3 = j - len_p3 # This is the starting index of the p3 match within s

            # Check if p3 occurs starting at end_p3 in s
            is_p3_match_at_end_p3 = False
            if len_p3 > 0:
                # Check if starts_p3 contains end_p3 using binary search
                idx_p3 = bisect.bisect_left(starts_p3, end_p3)
                if idx_p3 < len(starts_p3) and starts_p3[idx_p3] == end_p3:
                    is_p3_match_at_end_p3 = True
            else: # p3 is empty, it matches at any valid index end_p3 (from 0 to len(s))
                 # end_p3 = j - 0 = j. j iterates from min_core_len >= 0 up to len(s).
                 # So 0 <= end_p3 <= len(s) is always true within the loop.
                 is_p3_match_at_end_p3 = True
            
            if not is_p3_match_at_end_p3:
                continue # p3 does not match at the required position for this j

            # Find the largest index k in starts_p2 such that k <= end_p3 - len_p2
            # This is the latest possible start position for p2, constrained by the end of p3 match.
            limit_k = end_p3 - len_p2
            
            k_opt = -1 # Represents no valid k found

            if len_p2 > 0:
                # Find the index of the largest element <= limit_k in starts_p2
                idx_k = bisect.bisect_right(starts_p2, limit_k) - 1
                if idx_k >= 0:
                    k_opt = starts_p2[idx_k]
            else: # p2 is empty. It can match at any index k from 0 to len(s).
                 # Find the largest such index <= limit_k.
                 if limit_k >= 0: # k must be >= 0
                     k_opt = min(limit_k, len(s)) # Indices go up to len(s) for empty string matches

            if k_opt == -1: # No valid k found for this j and its derived end_p3
                 continue

            # Find the largest index start in starts_p1 such that start <= k_opt - len_p1
            # This is the latest possible start position for p1, constrained by the chosen k_opt.
            limit_start = k_opt - len_p1

            start_opt = -1 # Represents no valid start found

            if len_p1 > 0:
                # Find the index of the largest element <= limit_start in starts_p1
                idx_start = bisect.bisect_right(starts_p1, limit_start) - 1
                if idx_start >= 0:
                    start_opt = starts_p1[idx_start]
            else: # p1 is empty. It can match at any index start from 0 to len(s).
                 # Find the largest such index <= limit_start.
                 if limit_start >= 0: # start must be >= 0
                     start_opt = min(limit_start, len(s)) # Indices go up to len(s) for empty string matches

            if start_opt == -1: # No valid start found for the chosen k_opt
                 continue

            # Found valid start_opt, k_opt, end_p3 (derived from j).
            # The substring starts at start_opt and ends at j (exclusive).
            # Length is j - start_opt.
            
            # The conditions start + len_p1 <= k and k + len_p2 <= end_p3 must hold.
            # We selected k_opt <= end_p3 - len_p2, so k_opt + len_p2 <= end_p3 holds.
            # We selected start_opt <= k_opt - len_p1, so start_opt + len_p1 <= k_opt holds.
            # The indices start_opt, k_opt, end_p3 are valid start indices (>=0 and <= len(s)
            # when referring to empty matches, or within bounds for non-empty matches).

            current_len = j - start_opt
            min_len = min(min_len, current_len)

        # After iterating through all possible j
        if min_len == float('inf'):
            return -1
        else:
            return min_len