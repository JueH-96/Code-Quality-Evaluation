import itertools

class Solution:
  def maxActiveSectionsAfterTrade(self, s: str) -> int:
    n = len(s)
    initial_ones = s.count('1')

    # Augmented string t
    t_str = '1' + s + '1'

    # Parse t into blocks of consecutive characters
    # Example: t_str = "10011101" -> parsed_blocks = [('1',1), ('0',2), ('1',3), ('0',1), ('1',1)]
    parsed_blocks = []
    if not t_str: # Should not happen based on constraints (n>=1 means t_str is at least "11")
        return 0 # Or handle as error
        
    # Manual implementation of itertools.groupby logic
    # (itertools.groupby might be slightly cleaner but this is equivalent)
    current_char_type = t_str[0]
    current_char_count = 0
    for char_val in t_str:
        if char_val == current_char_type:
            current_char_count += 1
        else:
            parsed_blocks.append((current_char_type, current_char_count))
            current_char_type = char_val
            current_char_count = 1
    parsed_blocks.append((current_char_type, current_char_count)) # Add the last block

    # Separate lengths of '1'-blocks (lenA) and '0'-blocks (lenB)
    # lenA: lengths of A_0, A_1, ..., A_{k+1}
    # lenB: lengths of B_0, B_1, ..., B_k
    lenA = []
    lenB = []
    # parsed_blocks always starts with a '1' block because t_str[0] == '1'
    for i, (char_val, length) in enumerate(parsed_blocks):
        if char_val == '1':
            lenA.append(length)
        else:
            lenB.append(length)
    
    # A_x (candidate for B_1) must be lenA[x_idx_A] where 1 <= x_idx_A <= len(lenA)-2.
    # This requires len(lenA) to be at least 3 (A_0, A_1, A_2).
    # If len(lenA) < 3, no A_x (like A_1) can be chosen for step 1.
    if len(lenA) < 3: 
        return initial_ones

    num_b_blocks = len(lenB) # This is k+1 in `A_0 B_0 ... B_k A_{k+1}`

    # If len(lenA) >= 3, then lenB must have at least 2 elements.
    # Example: t="10101", s="010". lenA=[1,1,1], lenB=[1,1]. num_b_blocks=2.
    # The loop for x_idx_A will run for x_idx_A=1.
    # It will access lenB[0] and lenB[1]. This is valid.

    prefix_max_lenB = [0] * num_b_blocks
    prefix_max_lenB[0] = lenB[0]
    for i in range(1, num_b_blocks):
        prefix_max_lenB[i] = max(prefix_max_lenB[i-1], lenB[i])

    suffix_max_lenB = [0] * num_b_blocks
    suffix_max_lenB[num_b_blocks-1] = lenB[num_b_blocks-1]
    for i in range(num_b_blocks-2, -1, -1):
        suffix_max_lenB[i] = max(suffix_max_lenB[i+1], lenB[i])
    
    max_gain = 0 # If no beneficial trade, gain is 0 (perform no trade)

    # Iterate x_idx_A from 1 to len(lenA)-2 (inclusive).
    # This corresponds to choosing A_1, A_2, ..., A_k from the model.
    for x_idx_A in range(1, len(lenA) - 1):
        L1 = lenA[x_idx_A] # Length of the '1'-block chosen for step 1 (A_{x_idx_A})

        # Gain option 1: merged block B_{x_idx_A-1} A_{x_idx_A} B_{x_idx_A}
        # The '0'-block chosen is B_{x_idx_A-1} + A_{x_idx_A} (now '0's) + B_{x_idx_A}.
        # Its length L0 = lenB[x_idx_A-1] + L1 + lenB[x_idx_A].
        # Gain = L0 - L1 = lenB[x_idx_A-1] + lenB[x_idx_A].
        gain1 = lenB[x_idx_A - 1] + lenB[x_idx_A]
        
        # Gain option 2: choose an existing B_y block far from A_{x_idx_A} for step 2
        max_other_L0 = 0
        # Max from B_0, ..., B_{x_idx_A-2} (these are lenB[0]...lenB[x_idx_A-2])
        idx_b_before = x_idx_A - 2 # index for lenB, corresponds to B_{x_idx_A-2}
        if idx_b_before >= 0:
            max_other_L0 = max(max_other_L0, prefix_max_lenB[idx_b_before])
        
        # Max from B_{x_idx_A+1}, ..., B_k (these are lenB[x_idx_A+1]...lenB[num_b_blocks-1])
        idx_b_after = x_idx_A + 1 # index for lenB, corresponds to B_{x_idx_A+1}
        if idx_b_after < num_b_blocks:
            max_other_L0 = max(max_other_L0, suffix_max_lenB[idx_b_after])
        
        gain2 = max_other_L0 - L1
        
        current_max_gain_for_this_Ax = max(gain1, gain2)
        max_gain = max(max_gain, current_max_gain_for_this_Ax)
            
    return initial_ones + max_gain