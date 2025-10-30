import sys
from bisect import bisect_left, bisect_right

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:

        def find_all(text, pattern):
            # Custom find_all that handles empty pattern matching at all positions
            if not pattern:
                # Empty string matches at every position (including just after the last character)
                return list(range(len(text) + 1))
            indices = []
            i = -1
            while True:
                # Search starting from the character *after* the last match
                i = text.find(pattern, i + 1)
                if i == -1:
                    break
                indices.append(i)
            return indices

        # Find the positions of the two '*'
        # The problem guarantees exactly two '*'
        idx1 = p.find('*')
        idx2 = p.find('*', idx1 + 1)

        # Split the pattern into three parts based on the '*'
        part1 = p[:idx1]
        part2 = p[idx1 + 1 : idx2]
        part3 = p[idx2 + 1 :]

        # Find all starting indices of occurrences of part1, part2, part3 in s
        i_list = find_all(s, part1)
        k_list = find_all(s, part2)
        m_list = find_all(s, part3)

        # If part1 or part3 are not found in s (and are not empty), no match is possible.
        # Note: find_all returns [0, ..., len(s)] for empty pattern, which is never empty.
        if part1 and not i_list:
            return -1
        if part3 and not m_list:
            return -1
        
        # If part2 is not found (and is not empty), a match is only possible if
        # the required middle section in s (between end of part1 and start of part3)
        # is empty AND part2 is empty. Our logic naturally handles this.
        # If part2 is non-empty but k_list is empty, the suffix_min_k_plus_len_p2
        # computation will result in sys.maxsize, correctly indicating no valid k exists.

        # Precompute suffix minimums for k_list + len(part2)
        # suffix_min_k_plus_len_p2[j] = min(k_list[l] + len(part2) for l in range(j, len(k_list)))
        # This helps in finding the earliest possible start position for the part
        # corresponding to the second '*' and part3, given that part2 must start at
        # or after some index k.
        len_k_list = len(k_list)
        len_p2 = len(part2)
        len_p3 = len(part3)
        
        suffix_min_k_plus_len_p2 = [sys.maxsize] * (len_k_list + 1)
        # Iterate backwards to compute suffix minimums
        for j in range(len_k_list - 1, -1, -1):
            # If part2 starts at k_list[j], the earliest possible start for part3 is k_list[j] + len_p2.
            # We want the minimum such value considering all k >= k_list[j].
            current_k_plus_len_p2 = k_list[j] + len_p2
            suffix_min_k_plus_len_p2[j] = min(current_k_plus_len_p2, suffix_min_k_plus_len_p2[j+1])

        min_len = sys.maxsize
        k_ptr = 0 # Pointer to the first element in k_list that is >= mid_start_s

        # Iterate through all possible start indices of the matching substring
        # A matching substring must start with an occurrence of part1
        for i in i_list:
            # The part corresponding to the first '*' and part2 must start after part1 ends.
            # This is the start of the conceptual "middle" section of the s substring.
            mid_start_s = i + len(part1)

            # Find the first occurrence of part2 at or after mid_start_s.
            # Advance the k_ptr to point to the first k in k_list >= mid_start_s.
            # Since i_list is sorted, mid_start_s is non-decreasing, allowing us to advance k_ptr efficiently.
            while k_ptr < len_k_list and k_list[k_ptr] < mid_start_s:
                k_ptr += 1

            # If k_ptr reaches the end of k_list, it means there are no occurrences of part2
            # at or after mid_start_s for this i (or any subsequent i).
            # In this case, any requirement depending on finding such a k will result in infinity.
            # The suffix_min_k_plus_len_p2 at this point (suffix_min_k_plus_len_p2[len_k_list] = sys.maxsize) handles this.

            # The minimum required start index for part3 (m) must satisfy two conditions:
            # 1. The "middle" section s[mid_start_s : m] must exist or be empty, so m >= mid_start_s.
            # 2. Part2 must be a substring of s[mid_start_s : m]. This means there must exist
            #    some k in k_list such that mid_start_s <= k and k + len(part2) <= m.
            #    This second condition is equivalent to m >= min(k + len(part2) for k in k_list if k >= mid_start_s).
            #    The value min(k + len(part2) for k in k_list if k >= mid_start_s) is precisely
            #    suffix_min_k_plus_len_p2[k_ptr].

            # So, m must be >= mid_start_s AND m >= suffix_min_k_plus_len_p2[k_ptr].
            # This means m must be >= max(mid_start_s, suffix_min_k_plus_len_p2[k_ptr]).
            
            required_m_start = max(mid_start_s, suffix_min_k_plus_len_p2[k_ptr])
            
            # If required_m_start is infinity (because suffix_min_k_plus_len_p2[k_ptr] was inf, meaning no k >= mid_start_s),
            # or if required_m_start is beyond the bounds of s (a pattern part cannot start after len(s)),
            # then no possible starting index m for part3 exists for this i.
            if required_m_start > len(s) or required_m_start == sys.maxsize:
                 continue # This 'i' cannot start a valid match

            # Find the earliest occurrence of part3 (m) in m_list that starts at or after required_m_start.
            # We use binary search (bisect_left) on the sorted m_list.
            m_idx = bisect_left(m_list, required_m_start)

            # If we found such an m (i.e., m_idx is within the bounds of m_list)
            if m_idx < len(m_list):
                found_m = m_list[m_idx]

                # We found a valid match:
                # It starts at i (s[i : i + len(part1)] == part1 is guaranteed by i_list)
                # Part3 starts at found_m (s[found_m : found_m + len(part3)] == part3 is guaranteed by m_list)
                # The middle part s[i + len(part1) : found_m] is guaranteed to contain part2
                # because found_m >= required_m_start, and required_m_start was derived
                # to ensure m is large enough and after mid_start_s to contain some valid k.

                # The matching substring in s starts at index i and ends at index found_m + len(part3) - 1.
                # The Python slice s[i : found_m + len(part3)] represents this substring.
                # The length is (end index + 1) - (start index) = (found_m + len(part3)) - i
                current_len = (found_m + len_p3) - i
                min_len = min(min_len, current_len)

        # If min_len is still sys.maxsize, it means no valid match was found for any starting position i.
        return min_len if min_len != sys.maxsize else -1