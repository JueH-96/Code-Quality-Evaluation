import re
import bisect
import math

class Solution:
  def shortestMatchingSubstring(self, s: str, p: str) -> int:
    idx1 = p.find('*')
    # The second '*' must be after the first one.
    idx2 = p.find('*', idx1 + 1)

    P1_str = p[0:idx1]
    P2_str = p[idx1+1:idx2]
    P3_str = p[idx2+1:]

    # Helper to find all occurrences using re.finditer for O(N+M) performance.
    # re.finditer handles empty pattern_str correctly, returning matches at all positions.
    def find_all_occurrences(text: str, pattern_str: str) -> list[int]:
        # For this problem, pattern_str contains only lowercase English letters,
        # so re.escape() is not strictly needed but is good practice generally.
        # If pattern_str is empty, re.finditer("", text) yields matches at indices 0, 1, ..., len(text).
        return [match.start() for match in re.finditer(re.escape(pattern_str), text)]

    starts1 = find_all_occurrences(s, P1_str)
    starts2 = find_all_occurrences(s, P2_str)
    starts3 = find_all_occurrences(s, P3_str)
    
    min_total_len = math.inf

    len_P1 = len(P1_str)
    len_P2 = len(P2_str)
    len_P3 = len(P3_str)

    # If P2_str is non-empty and not found, starts2 will be empty. 
    # The loop 'for j_start in starts2:' will not run, and min_total_len remains inf.
    # This correctly leads to returning -1 if no match is possible.
    
    for j_start in starts2:
        # Condition for i_start: i_start + len_P1 <= j_start  =>  i_start <= j_start - len_P1
        # We want the largest such i_start to minimize length ((k_end) - i_start).
        target_i = j_start - len_P1
        
        # bisect_right returns idx_i s.t. all elements in starts1[:idx_i] are <= target_i
        idx_i = bisect.bisect_right(starts1, target_i)
        
        if idx_i == 0: # No element in starts1 is <= target_i
            continue 
        actual_i_start = starts1[idx_i - 1] # Largest element in starts1 <= target_i

        # Condition for k_start: j_start + len_P2 <= k_start
        # We want the smallest such k_start to minimize length (k_start + len_P3) - i_start.
        target_k = j_start + len_P2
        
        # bisect_left returns idx_k s.t. all elements in starts3[idx_k:] are >= target_k
        idx_k = bisect.bisect_left(starts3, target_k)

        if idx_k == len(starts3): # All elements in starts3 are < target_k
            continue
        actual_k_start = starts3[idx_k] # Smallest element in starts3 >= target_k
        
        # A valid combination (actual_i_start, j_start, actual_k_start) is found.
        # The substring in s starts at actual_i_start.
        # It ends at (actual_k_start + len_P3 - 1).
        # The length is (actual_k_start + len_P3) - actual_i_start.
        current_len = (actual_k_start + len_P3) - actual_i_start
        min_total_len = min(min_total_len, current_len)

    if min_total_len == math.inf:
        return -1
    else:
        return min_total_len