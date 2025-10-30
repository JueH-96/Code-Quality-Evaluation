class Solution:
  def countCompleteSubstrings(self, word: str, k: int) -> int:
    total_ans = 0
    n = len(word)
    
    i = 0
    while i < n:
        # Start of a new segment
        start_of_segment = i
        
        # Find end of current segment
        j = i
        while j + 1 < n and abs(ord(word[j+1]) - ord(word[j])) <= 2:
            j += 1
        # Segment is word[start_of_segment : j+1]
        # (i.e., word characters from index start_of_segment to j, inclusive)
        
        segment_str = word[start_of_segment : j+1]
        
        # Process this segment
        total_ans += self._solve_segment(segment_str, k)
        
        # Move to the start of the next potential segment
        i = j + 1
            
    return total_ans

  def _solve_segment(self, s_str: str, k_val: int) -> int:
    ans_segment = 0
    segment_len = len(s_str)

    # Iterate over possible number of distinct characters (m_distinct_target)
    for m_distinct_target in range(1, 27): # m_distinct_target from 1 to 26
        window_len = m_distinct_target * k_val
        
        if window_len == 0: # Should not happen given k_val >= 1
             continue
        if window_len > segment_len:
            # Window too long for this segment, and will only get longer for larger m_distinct_target
            break 

        # counts: frequency of each char 'a' through 'z'
        counts = [0] * 26
        # current_distinct_chars: Number of characters with count > 0 in window
        current_distinct_chars = 0 
        # chars_at_freq_k: Number of characters with count == k_val in window
        chars_at_freq_k = 0      
        
        # Initialize first window: s_str[0 ... window_len-1]
        # This loop runs window_len times, for indices 0 to window_len-1
        for idx in range(window_len):
            char_val = s_str[idx]
            char_code = ord(char_val) - ord('a')
            
            # === Add char_val to window ===
            old_count = counts[char_code]
            counts[char_code] += 1
            new_count = counts[char_code]

            if old_count == 0: # Character appears for the first time in window
                current_distinct_chars += 1
            
            # Update chars_at_freq_k based on count changes
            if new_count == k_val: 
                chars_at_freq_k += 1
            # Important: check old_count as well. If old_count was k_val, and new_count is not,
            # then this character is no longer "at frequency k".
            if old_count == k_val: 
                chars_at_freq_k -= 1
            # === End Add char_val ===

        # Check initial window s_str[0 ... window_len-1]
        if current_distinct_chars == m_distinct_target and chars_at_freq_k == m_distinct_target:
            ans_segment += 1
        
        # Slide window across the segment
        # Window start index goes from 1 up to segment_len - window_len
        # New window is s_str[start_idx ... start_idx + window_len - 1]
        for start_idx in range(1, segment_len - window_len + 1):
            # Character to remove from left: s_str[start_idx - 1]
            char_out_val = s_str[start_idx - 1]
            char_out_code = ord(char_out_val) - ord('a')
            
            # === Remove char_out_val from window ===
            old_count_out = counts[char_out_code]
            counts[char_out_code] -= 1
            new_count_out = counts[char_out_code]

            if new_count_out == 0: # Character count becomes 0
                current_distinct_chars -= 1
            
            # Update chars_at_freq_k based on count changes for char_out
            if new_count_out == k_val:
                chars_at_freq_k += 1
            if old_count_out == k_val:
                chars_at_freq_k -= 1
            # === End Remove char_out_val ===

            # Character to add to right: s_str[start_idx + window_len - 1]
            char_in_val = s_str[start_idx + window_len - 1]
            char_in_code = ord(char_in_val) - ord('a')

            # === Add char_in_val to window ===
            old_count_in = counts[char_in_code]
            counts[char_in_code] += 1
            new_count_in = counts[char_in_code]

            if old_count_in == 0: # Character appears for the first time in window
                current_distinct_chars += 1
            
            # Update chars_at_freq_k based on count changes for char_in
            if new_count_in == k_val:
                chars_at_freq_k += 1
            if old_count_in == k_val:
                chars_at_freq_k -= 1
            # === End Add char_in_val ===
            
            # Check current window s_str[start_idx ... start_idx + window_len - 1]
            if current_distinct_chars == m_distinct_target and chars_at_freq_k == m_distinct_target:
                ans_segment += 1
            
    return ans_segment