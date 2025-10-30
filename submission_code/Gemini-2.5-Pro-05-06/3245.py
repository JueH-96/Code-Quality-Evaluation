from typing import List

class Solution:
  def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
    n = len(s)
    len_a = len(a)
    len_b = len(b)

    # Step 1: Find all starting indices of 'a' in 's'.
    indices_a = []
    # The loop `range(n - len_a + 1)` correctly handles cases where len_a > n
    # (the range becomes empty, so indices_a remains empty).
    for i in range(n - len_a + 1):
        # s.startswith(substring, offset) is efficient for this.
        if s.startswith(a, i):
            indices_a.append(i)
    
    # Step 2: Find all starting indices of 'b' in 's'.
    indices_b = []
    for i in range(n - len_b + 1):
        if s.startswith(b, i):
            indices_b.append(i)
    
    # Step 3: If 'a' or 'b' does not appear in 's', no beautiful indices can exist.
    if not indices_a or not indices_b:
        return []

    # Step 4: Iterate through indices of 'a' and find if a valid 'b' index exists.
    # Use a two-pointer/sliding window approach for efficiency.
    result = []
    idx_b = 0 # Pointer for current position in indices_b.
    num_b_indices = len(indices_b) # Cache length of indices_b for performance.

    for i_val in indices_a:
        # Advance idx_b: We need to find a j_val in indices_b such that:
        # i_val - k <= j_val <= i_val + k
        #
        # First, move idx_b forward so that indices_b[idx_b] >= i_val - k.
        # Any indices_b[j] where j < idx_b (after this loop) are too small
        # (i.e., < i_val - k). They would also be too small for any future i_val'
        # from indices_a (since future i_val' >= i_val, thus i_val' - k >= i_val - k).
        while idx_b < num_b_indices and indices_b[idx_b] < i_val - k:
            idx_b += 1
        
        # If idx_b has moved past the end of indices_b, it means all remaining
        # occurrences of 'b' are too small for the current i_val (and any subsequent ones),
        # or indices_b was exhausted. So, no more beautiful indices can be found.
        if idx_b == num_b_indices:
            break 

        # Now, indices_b[idx_b] is the smallest occurrence of 'b' (at or after current idx_b)
        # that is >= i_val - k.
        # We must also check if this occurrence is not too large: indices_b[idx_b] <= i_val + k.
        # If both conditions hold (i_val - k <= indices_b[idx_b] <= i_val + k),
        # then i_val is a beautiful index.
        if indices_b[idx_b] <= i_val + k:
            result.append(i_val)
            
    return result