import math

class Solution:
  """
  Finds the length of the shortest substring in s that matches pattern p,
  where p contains exactly two '*' characters acting as wildcards.
  The '*' matches any sequence of zero or more characters.
  """
  def shortestMatchingSubstring(self, s: str, p: str) -> int:
    """
    Calculates the length of the shortest substring of s that matches pattern p.

    Args:
      s: The text string.
      p: The pattern string containing exactly two '*'.

    Returns:
      The length of the shortest matching substring, or -1 if no such substring exists.
    """
    n = len(s)
    
    # Find the indices of the two '*' characters in p.
    # The constraints guarantee exactly two '*' exist.
    star_indices = [i for i, char in enumerate(p) if char == '*']
    star1_idx, star2_idx = star_indices[0], star_indices[1]
    
    # Parse p into three parts: prefix, middle, and suffix based on '*' positions.
    prefix = p[:star1_idx]
    middle = p[star1_idx + 1 : star2_idx]
    suffix = p[star2_idx + 1 :]

    # Calculate the lengths of the three parts.
    lp = len(prefix)
    lm = len(middle)
    ls = len(suffix)

    # Helper function to find all start indices of a non-empty pattern in a text.
    # Uses Python's built-in string.find for efficiency.
    def findAllOccurrences(text, pattern):
        indices = []
        start = 0
        # Loop to find all occurrences
        while True:
            # Find the next occurrence at or after 'start' index
            idx = text.find(pattern, start)
            if idx == -1:
                # No more occurrences found
                break
            indices.append(idx)
            # Move the start position for the next search. 
            # start = idx + 1 ensures we find all occurrences, including overlapping ones.
            start = idx + 1 
        return indices

    # Find all occurrences of non-empty prefix, middle, and suffix parts in s.
    # If a part is empty, the corresponding list will be empty, handled by length checks later.
    P = findAllOccurrences(s, prefix) if lp > 0 else []
    M = findAllOccurrences(s, middle) if lm > 0 else []
    S = findAllOccurrences(s, suffix) if ls > 0 else []

    # If a required non-empty part (prefix, middle, or suffix) is not found in s, 
    # then no match is possible according to the pattern structure.
    if lp > 0 and not P: return -1
    if lm > 0 and not M: return -1
    if ls > 0 and not S: return -1
    
    # Initialize minimum length found so far to infinity.
    min_len = float('inf')

    # Handle the different cases based on which parts of p (prefix, middle, suffix) are empty.
    # These cases represent patterns like "**", "**suffix", "prefix**", "*middle*", etc.

    if lp == 0 and lm == 0 and ls == 0: # Case: p = "**"
        # Matches the empty substring. Shortest length is 0.
        return 0
    
    elif lp == 0 and lm == 0: # Case: p = "**suffix" (ls > 0)
        # Shortest match is the suffix itself. Length ls.
        # Requires suffix to exist in s, which is guaranteed by the check `(ls > 0 and not S)` above.
        return ls 
            
    elif lm == 0 and ls == 0: # Case: p = "prefix**" (lp > 0)
        # Shortest match is the prefix itself. Length lp.
        # Requires prefix to exist in s.
        return lp 

    elif lp == 0 and ls == 0: # Case: p = "*middle*" (lm > 0)
        # Shortest match is the middle itself. Length lm.
        # Requires middle to exist in s.
        return lm 

    # Cases where exactly one part is empty: Use two-pointer approach optimized for each scenario.
    
    elif lp == 0: # Case: p = "*middle*suffix" (lm > 0, ls > 0)
        # We need a middle starting at k (k in M) and a suffix starting at l (l in S)
        # such that the suffix starts after the middle ends (l >= k + lm).
        # The match starts conceptually right before middle (i=k, since prefix is empty).
        # We want to minimize length = (l + ls) - i = l + ls - k.
        ptr_s = 0 # Pointer for the sorted list S of suffix start indices.
        for k in M: # Iterate through all possible start indices k for the middle part.
            target_l = k + lm # The suffix must start at or after this index.
            # Advance ptr_s to find the first suffix start index l >= target_l.
            # Since M and S are sorted, ptr_s only moves forward.
            while ptr_s < len(S) and S[ptr_s] < target_l:
                ptr_s += 1
            
            if ptr_s < len(S): # If a valid suffix start index l is found
                l_min = S[ptr_s] # This is the earliest suffix start for the current k.
                current_len = l_min + ls - k # Calculate the length of this match.
                min_len = min(min_len, current_len) # Update the overall minimum length.
    
    elif lm == 0: # Case: p = "prefix**suffix" (lp > 0, ls > 0)
        # We need a prefix starting at i (i in P) and a suffix starting at l (l in S)
        # such that the suffix can start after the prefix ends (l >= i + lp).
        # This condition ensures a middle part (even empty) can fit between them.
        # We want to minimize length = (l + ls) - i.
        ptr_p = 0 # Pointer for the sorted list P of prefix start indices.
        last_valid_i = -1 # Tracks the latest valid prefix start index found so far for the current l.
        for l in S: # Iterate through all possible start indices l for the suffix part.
             target_i_end = l # The prefix must end at or before index l (i.e., i + lp <= l).
             # Advance ptr_p to find the latest prefix start index i satisfying the condition.
             while ptr_p < len(P) and P[ptr_p] + lp <= target_i_end:
                 last_valid_i = P[ptr_p] # Update the latest valid i found.
                 ptr_p += 1
             
             if last_valid_i != -1: # If a valid prefix start was found for this suffix start l
                 current_len = l + ls - last_valid_i # Calculate length using the latest prefix start.
                 min_len = min(min_len, current_len) # Update minimum length.

    elif ls == 0: # Case: p = "prefix*middle*" (lp > 0, lm > 0)
        # We need a prefix starting at i (i in P) and a middle starting at k (k in M)
        # such that the middle starts after the prefix ends (k >= i + lp).
        # The match ends right after the middle part since the suffix is empty.
        # The conceptual suffix start l is k + lm.
        # We want to minimize length = (l + ls) - i = (k + lm + 0) - i.
        ptr_p = 0 # Pointer for the sorted list P.
        last_valid_i = -1 # Tracks the latest valid prefix start index for the current k.
        for k in M: # Iterate through all possible start indices k for the middle part.
            target_i_end = k # The prefix must end at or before index k (i.e., i + lp <= k).
            # Advance ptr_p to find the latest prefix start index i satisfying the condition.
            while ptr_p < len(P) and P[ptr_p] + lp <= target_i_end:
                last_valid_i = P[ptr_p]
                ptr_p += 1
            
            if last_valid_i != -1: # If a valid prefix start was found for this middle start k
                current_len = (k + lm) - last_valid_i # Calculate length.
                min_len = min(min_len, current_len) # Update minimum length.

    # Case where all parts (prefix, middle, suffix) are non-empty (lp > 0, lm > 0, ls > 0)
    else: 
        ptr_p = 0 # Pointer for P list.
        ptr_s = 0 # Pointer for S list.
        last_valid_i = -1 # Tracks the latest valid prefix start index for the current k.
        for k in M: # Iterate through all possible start indices k for the middle part.
            # Find the latest prefix start index 'i' such that the prefix ends before 'k' starts (i + lp <= k).
            target_i_end = k
            while ptr_p < len(P) and P[ptr_p] + lp <= target_i_end:
                last_valid_i = P[ptr_p]
                ptr_p += 1
            
            if last_valid_i == -1:
                # If no prefix ends early enough for this middle start k, try the next k.
                # This can happen if the first prefix starts after the current k.
                continue 

            # Find the earliest suffix start index 'l' such that the suffix starts after 'k' ends (l >= k + lm).
            target_l = k + lm
            # Advance ptr_s to find the first suffix start index l >= target_l.
            while ptr_s < len(S) and S[ptr_s] < target_l:
                ptr_s += 1
            
            if ptr_s == len(S):
                # If we have exhausted all suffix occurrences, no further matches are possible
                # for the current or any subsequent middle start k, because k increases.
                break 

            # A valid combination (i, k, l) is found: (last_valid_i, k, S[ptr_s]).
            # Calculate the length of the substring for this combination.
            l_min = S[ptr_s] # The earliest suffix start index for this k.
            current_len = l_min + ls - last_valid_i # Length = (end of suffix) - (start of prefix)
            min_len = min(min_len, current_len) # Update the overall minimum length.
            
    # After checking all possibilities, return the minimum length found.
    # If min_len is still infinity, it means no match was found.
    return min_len if min_len != float('inf') else -1