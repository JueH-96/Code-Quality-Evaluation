class Solution:
  def _get_char_val_recursive(self, idx: int, ops_count: int) -> int:
    """
    Helper function to find the character value (0-25 for 'a'-'z') at a given 
    0-indexed position `idx` in the string generated after `ops_count` operations (W_{ops_count}).
    
    String construction:
    W_0 = "a"
    W_n = W_{n-1} + transform(W_{n-1})
    
    The character at `idx` in W_{ops_count}:
    - If `idx` is in the first half (i.e., part of W_{ops_count-1}):
      It's the same as the character at `idx` in W_{ops_count-1}.
    - If `idx` is in the second half (i.e., part of transform(W_{ops_count-1})):
      It's the transformed version of the character at `idx - len_first_half`
      in W_{ops_count-1}. `len_first_half` is the length of W_{ops_count-1}.
    """
    # Base case: W_0 (0 operations) is "a".
    # For W_0, `idx` must be 0. The character is 'a', which has value 0.
    if ops_count == 0:
      return 0 
    
    # Length of W_{ops_count-1}. This string forms the first half of W_{ops_count}.
    # Length of W_{ops_count-1} is 2^(ops_count - 1).
    len_first_half = 1 << (ops_count - 1) 

    if idx < len_first_half:
      # Character is in the W_{ops_count-1} part (first half).
      # Recursively find its value from the string of the previous iteration.
      return self._get_char_val_recursive(idx, ops_count - 1)
    else:
      # Character is in the transform(W_{ops_count-1}) part (second half).
      # Find the value of the character from which this one was transformed.
      # This character was at index `idx - len_first_half` in W_{ops_count-1}.
      val_from_prev = self._get_char_val_recursive(idx - len_first_half, ops_count - 1)
      # Transform it: increment value by 1, wrap around 'z' to 'a' (modulo 26).
      return (val_from_prev + 1) % 26
      
  def kthCharacter(self, k: int) -> str:
    """
    Calculates the k-th character (1-indexed) in the generated string.
    """
    # Convert k from 1-indexed to 0-indexed for internal calculations.
    k_zero_indexed = k - 1
    
    # Determine `m_ops`: the number of operations after which the string is W_{m_ops}.
    # W_{m_ops} has length 2^{m_ops}.
    # The target character at `k_zero_indexed` must be within W_{m_ops}, meaning
    # `k_zero_indexed < 2^{m_ops}`.
    # `m_ops` is the smallest integer satisfying this.
    # This value is equivalent to the number of bits required to represent `k_zero_indexed`
    # in binary (e.g., if k_zero_indexed=4 (100_2), m_ops=3).
    # For k_zero_indexed=0, (0).bit_length() is 0, so m_ops=0. This is correct (W_0).
    m_ops = k_zero_indexed.bit_length()
            
    # Get the integer value (0-25) of the character using the recursive helper.
    char_val = self._get_char_val_recursive(k_zero_indexed, m_ops)
    
    # Convert the integer value (0-25) back to its corresponding character 'a'-'z'.
    return chr(ord('a') + char_val)