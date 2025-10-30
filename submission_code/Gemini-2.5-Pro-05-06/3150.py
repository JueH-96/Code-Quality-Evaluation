class Solution:
  def shortestBeautifulSubstring(self, s: str, k: int) -> str:
    n = len(s)
    
    # Step 1: Find all indices of '1's in s
    one_indices = []
    for i in range(n):
        if s[i] == '1':
            one_indices.append(i)
    
    num_total_ones = len(one_indices)
    
    # Step 2: If there are fewer than k ones in s, no beautiful substring exists
    if num_total_ones < k:
        return ""
        
    # Step 3: Initialize min_len and result_substring
    # min_len stores the length of the shortest beautiful substring found so far.
    # result_substring stores the lexicographically smallest among those with min_len.
    min_len = float('inf') 
    result_substring = ""  # Will be updated as loop guarantees at least one candidate

    # Step 4: Iterate through all possible groups of k '1's
    # A group consists of k consecutive '1's from the one_indices list.
    # The loop iterates from the first possible start of such a group up to the last.
    # The number of iterations is (num_total_ones - k + 1).
    for i in range(num_total_ones - k + 1):
        # Determine the start and end indices in the original string 's'
        # for the current candidate substring.
        # The current group of k '1's starts at index one_indices[i] in 's'.
        start_s_idx = one_indices[i]
        # It ends at index one_indices[i + k - 1] in 's'.
        end_s_idx = one_indices[i + k - 1]
        
        current_len = end_s_idx - start_s_idx + 1
        
        # Compare with the current shortest beautiful substring found
        if current_len < min_len:
            # This candidate is shorter, so it becomes the new best.
            min_len = current_len
            # Substring slicing: s[inclusive_start : exclusive_end]
            result_substring = s[start_s_idx : end_s_idx + 1]
        elif current_len == min_len:
            # This candidate has the same length as the current shortest.
            # We need to check if it's lexicographically smaller.
            candidate_substring = s[start_s_idx : end_s_idx + 1]
            if candidate_substring < result_substring:
                result_substring = candidate_substring
                
    # Step 5: Return the result
    return result_substring