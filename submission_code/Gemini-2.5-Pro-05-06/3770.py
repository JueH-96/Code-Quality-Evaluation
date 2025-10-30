class Solution:
  def generateString(self, str1: str, str2: str) -> str:
    n = len(str1)
    m = len(str2)
    L = n + m - 1

    if L == 0: # handles n=0 or m=0 cases if possible based on constraints (n,m >=1 here)
        return "" if n > 0 else "" # if n>0 implies m=0 (L=n-1), then invalid.

    res = [''] * L 
    is_fixed = [False] * L

    for i in range(n):
      if str1[i] == 'T':
        for j in range(m):
          char_idx = i + j
          if is_fixed[char_idx] and res[char_idx] != str2[j]:
            return ""
          res[char_idx] = str2[j]
          is_fixed[char_idx] = True

    for i in range(n):
      if str1[i] == 'F':
        all_chars_in_window_fixed = True
        # Check if substring res[i:i+m] matches str2
        # This check is only relevant if all characters in this window are fixed.
        current_substring_match_str2 = True
        for j in range(m):
          char_idx = i + j
          if not is_fixed[char_idx]:
            all_chars_in_window_fixed = False
            break 
          if res[char_idx] != str2[j]: # If any fixed char differs, it's not str2
            current_substring_match_str2 = False
            # no need to break here, just continue to check all_chars_in_window_fixed
        
        if all_chars_in_window_fixed and current_substring_match_str2:
          return "" 
            
    for k in range(L):
      if not is_fixed[k]:
        res[k] = 'a'

    prev_free_map = [-1] * L
    last_free = -1
    for k in range(L):
        prev_free_map[k] = last_free
        if not is_fixed[k]:
            last_free = k
    
    is_violated = [False] * n
    for i in range(n):
        if str1[i] == 'F':
            match = True
            for j in range(m):
                if res[i+j] != str2[j]:
                    match = False
                    break
            if match:
                is_violated[i] = True
    
    # Safety counter for iterations, related to L * 26 total char increments
    # Max total increments is sum over free positions of (ord('z') - ord('a')) = L * 25.
    # Each outer loop makes progress on this sum.
    for iteration_count in range(L * 26 + 5): 
      i_offending = -1
      for i_check in range(n):
        if is_violated[i_check]: # str1[i_check] must be 'F'
          i_offending = i_check
          break
      
      if i_offending == -1:
        return "".join(res)
          
      # Determine k_mod: rightmost non-fixed char in window S_i_offending
      k_mod_in_error_window = -1 # Actual character index in `res`
      for k_check in range(i_offending + m - 1, i_offending - 1, -1):
        if not is_fixed[k_check]:
          k_mod_in_error_window = k_check
          break
      
      # This check for k_mod_in_error_window == -1 should ideally be caught by step 2
      # if all fixed and matches str2. If not all fixed, k_mod must exist.
      # If it's -1 here, implies an issue.
      if k_mod_in_error_window == -1: return ""


      # This k_mod will be the starting point for modification.
      # It might shift left due to carries.
      current_k_to_modify = k_mod_in_error_window
      
      # Store range of characters that actually change for updating affected windows later
      # min_idx_changed initially is k_mod_in_error_window, max is k_mod_in_error_window
      # It will expand if carries occur.
      min_idx_changed_in_res = k_mod_in_error_window 
      max_idx_changed_in_res = k_mod_in_error_window


      while True: # Inner loop for char increment and carry
        current_char_val = ord(res[current_k_to_modify])
        current_char_val += 1
        res[current_k_to_modify] = chr(current_char_val)
        min_idx_changed_in_res = min(min_idx_changed_in_res, current_k_to_modify)


        if current_char_val > ord('z'):
          res[current_k_to_modify] = 'a' 
          current_k_to_modify = prev_free_map[current_k_to_modify]
          
          if current_k_to_modify < i_offending : 
            # Carry propagated out of the left side of S_i_offending window
            # This means S_i_offending cannot be fixed by changes within or to its right.
             return ""
          if current_k_to_modify == -1: # No more free spots to left to carry to.
             return ""
          # continue inner loop to increment new current_k_to_modify
        else: # No carry out of current_k_to_modify
            # Check if S_i_offending is now different from str2
            is_S_i_offending_now_matching_str2 = True
            for j_ptr in range(m):
                if res[i_offending + j_ptr] != str2[j_ptr]:
                    is_S_i_offending_now_matching_str2 = False
                    break
            
            if not is_S_i_offending_now_matching_str2:
              # S_i_offending is fixed. Update affected windows and break from inner loop.
              # The characters that changed are in [min_idx_changed_in_res, max_idx_changed_in_res]
              # Check windows S_p that overlap with this changed segment.
              # Window S_p = res[p...p+m-1]
              # Overlap if p+m-1 >= min_idx_changed_in_res AND p <= max_idx_changed_in_res
              start_p_check = max(0, min_idx_changed_in_res - m + 1)
              end_p_check = min(n -1 , max_idx_changed_in_res) # window starts at p, so p can be up to max_idx_changed_in_res

              for p_check in range(start_p_check, end_p_check + 1):
                  if str1[p_check] == 'F':
                      # Re-evaluate S_p_check
                      match = True
                      for j_ptr in range(m):
                          if res[p_check + j_ptr] != str2[j_ptr]:
                              match = False
                              break
                      is_violated[p_check] = match
              break # Break from inner char increment/carry loop
            # else: S_i_offending still matches str2. Current char change was not enough. Continue inner loop.
    
    return "" # Should be unreachable if logic is correct within iteration bound