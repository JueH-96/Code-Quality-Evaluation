class Solution:
  def minimumTimeToInitialState(self, word: str, k: int) -> int:
    n = len(word)
    
    # Calculate the maximum number of operations (time t) needed if no prefix/suffix match occurs earlier.
    # This happens when all original characters are shifted out (t*k >= n).
    # The smallest integer t satisfying this is ceil(n/k).
    # In integer arithmetic, for positive integers a, b: ceil(a/b) = (a + b - 1) // b.
    t_max_ops = (n + k - 1) // k
    
    # Iterate t from 1 up to t_max_ops.
    # The first t for which the condition is met will be the minimum time.
    for t in range(1, t_max_ops + 1):
        # Number of characters from the original word's prefix that have been effectively removed.
        num_chars_shifted_out = t * k
        
        # Length of the remaining part of the original word, W_0[t*k : n].
        # This part forms the prefix of the current word (before adding characters at step t).
        len_prefix_from_original = n - num_chars_shifted_out
        
        if len_prefix_from_original <= 0:
            # This means t*k >= n. All original characters of W_0 have been shifted out from the prefix.
            # The word now consists entirely of characters we added over t operations.
            # We can choose these added characters to make the word equal to W_0.
            # This condition is guaranteed to be met by t = t_max_ops if no earlier match.
            return t
        
        # If a part of the original word remains (len_prefix_from_original > 0):
        # This remaining part is W_0[t*k : n].
        # We need it to match the actual prefix of W_0, which is W_0[0 : len_prefix_from_original].
        # If they match, we can choose the characters added (at step t and previous steps)
        # to correctly form the suffix of W_0, thus restoring W_0.
        
        # Perform the comparison: W_0[t*k : n] vs W_0[0 : n-t*k]
        # Python slice word[num_chars_shifted_out:] is equivalent to word[num_chars_shifted_out : n].
        # Python slice word[0:len_prefix_from_original] is word[0 : n-t*k].
        
        # Note: The problem asks for `word` to revert to its initial state.
        # `word` in these comparisons refers to the initial state `W_0`.
        if word[num_chars_shifted_out:] == word[0:len_prefix_from_original]:
            return t
            
    # This part of the code should be unreachable because the loop for t up to t_max_ops
    # guarantees that len_prefix_from_original <= 0 will eventually be true,
    # causing a return from within the loop.
    # Adding a fallback return for completeness, though it logically won't be hit.
    return t_max_ops