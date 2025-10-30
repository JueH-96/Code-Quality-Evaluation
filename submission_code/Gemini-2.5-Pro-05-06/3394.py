class Solution:
  def minEnd(self, n: int, x: int) -> int:
    # Let m = n-1. We are looking for the m-th number (0-indexed)
    # in the sequence of values k_i that can be formed using only
    # bit positions where x has a 0. Call this k_m_target.
    # The final answer will be x | k_m_target.
    # The strategy is to initialize the answer to x, and then iterate
    # through bit positions. For each bit position that is 0 in x (a "free bit"),
    # we check the corresponding bit in m. If that bit in m is 1,
    # we set this free bit in our answer.

    # If n = 1, then m = 0. The m-th value (k_0_target) is 0.
    # The answer is x | 0 = x. The loop for m handles this correctly.
    # as m (idx_val in code) starts at 0, loop `while idx_val > 0` won't run.
    
    idx_val = n - 1 
    
    # ans_val initially holds x. We will OR it with bits that form k_m_target.
    ans_val = x 
    
    # current_bit_to_check_in_x iterates through powers of 2: 1, 2, 4, 8, ...
    # representing bit positions 0, 1, 2, 3, ...
    current_bit_to_check_in_x = 1 
    
    # Iterate as long as there are bits in idx_val to process (idx_val > 0).
    # When idx_val becomes 0, it means all its set bits (which determine which
    # powers of 2 (from free bit positions) to pick) have been processed.
    while idx_val > 0:
      # Check if the current bit position (represented by current_bit_to_check_in_x)
      # is a "free bit" (i.e., it's 0 in x).
      if (x & current_bit_to_check_in_x) == 0:
        # This position is a free bit. It corresponds to $2^{p_t}$ for some t
        # (t-th available free bit position).
        # We check if the t-th bit of (original n-1) is 1.
        # This is done by checking the LSB of the current idx_val.
        if (idx_val & 1) == 1:
          # LSB of idx_val is 1, so this free bit should be set in the result.
          ans_val = ans_val | current_bit_to_check_in_x
        
        # We have processed one bit of significance from idx_val.
        # (Effectively, we've used the LSB of idx_val to decide about $2^{p_t}$,
        # now we move to consider $2^{p_{t+1}}$ with the next bit of idx_val).
        idx_val = idx_val >> 1
      
      # Move to the next bit position.
      current_bit_to_check_in_x = current_bit_to_check_in_x << 1
      
      # Python's integers handle arbitrary size, so current_bit_to_check_in_x can grow large
      # without overflow. The loop terminates because idx_val decreases with each found free bit
      # and eventually reaches 0.
      # Max N, X are 10^8 (approx 2^27). idx_val has at most ~27 bits.
      # current_bit_to_check_in_x might go up to around 2^(27+27) = 2^54.
      
    return ans_val