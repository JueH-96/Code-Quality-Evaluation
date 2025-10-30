import sys
from typing import List

# Setting recursion depth is usually not needed for iterative solutions
# but can be included defensively if there were recursive aspects.
# sys.setrecursionlimit(2000) 

class Solution:
  """
  Solves the maximum number of removable characters problem using binary search.
  The problem asks for the maximum number `k` such that we can remove `k` characters
  from `source` at indices specified by the first `k` elements of `targetIndices`,
  and `pattern` still remains a subsequence of the modified `source`.

  The core idea is that the property "pattern is a subsequence after removing first k elements"
  is monotonic. If it holds for `k`, it holds for any `k' < k`. If it fails for `k`,
  it fails for any `k' > k`. This allows us to use binary search on `k`.
  """
  def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
    """
    Finds the maximum number k such that removing characters at indices 
    targetIndices[0]...targetIndices[k-1] leaves pattern as a subsequence of source.

    Args:
      source: The original string (length n).
      pattern: The pattern string which must remain a subsequence (length p_len).
      targetIndices: A sorted list of distinct indices in source that are candidates for removal (length m).

    Returns:
      The maximum integer k satisfying the condition.
    """
    n = len(source)
    p_len = len(pattern)
    m = len(targetIndices)

    def is_subsequence(k: int) -> bool:
        """
        Checks if `pattern` is a subsequence of `source` after removing k characters
        specified by the first k indices in `targetIndices`. 
        Time complexity: O(n + k). n for iterating through source, k for set creation.

        Args:
          k: The number of characters to remove (indices targetIndices[0]...targetIndices[k-1]).

        Returns:
          True if pattern is still a subsequence, False otherwise.
        """
        # Create a set of the first k indices from targetIndices to efficiently check for removal. 
        # This takes O(k) time.
        removed_indices = set(targetIndices[:k])
        
        p_ptr = 0 # Pointer for pattern string
        s_ptr = 0 # Pointer for source string
        
        # Use two pointers to check for subsequence property. This takes O(n) time.
        while s_ptr < n and p_ptr < p_len:
            # Check three conditions to advance the pattern pointer p_ptr:
            # 1. The current source index `s_ptr` must NOT be in the set of removed indices.
            # 2. The character `source[s_ptr]` must match the current pattern character `pattern[p_ptr]`.
            if s_ptr not in removed_indices and source[s_ptr] == pattern[p_ptr]:
                p_ptr += 1 # Found a character match, move to the next character in pattern.
            
            # Always advance the source pointer to check the next character in source.
            s_ptr += 1

        # After iterating through the source string, if `p_ptr` has reached the length of `pattern`,
        # it means we successfully found all characters of `pattern` in order.
        return p_ptr == p_len

    # Binary search on the number of removals k. The possible values for k are in the range [0, m].
    # The binary search performs O(log m) calls to the `is_subsequence` function.
    low = 0
    high = m
    ans = 0 # Initialize the answer (maximum k found so far) to 0.

    while low <= high:
        # Calculate the midpoint index to test.
        # Using `low + (high - low) // 2` avoids potential integer overflow compared to `(low + high) // 2`.
        mid = low + (high - low) // 2 
        
        # Check if removing the first 'mid' characters maintains the subsequence property.
        if is_subsequence(mid):
            # If `is_subsequence(mid)` is True, it means removing 'mid' characters is possible.
            # We record 'mid' as a potential answer because it satisfies the condition.
            # Since we want the maximum k, we try searching for a potentially larger k
            # in the upper half of the current range: [mid + 1, high].
            ans = mid  
            low = mid + 1 
        else:
            # If `is_subsequence(mid)` is False, removing 'mid' characters breaks the subsequence property.
            # Due to monotonicity, any k > mid will also fail.
            # Therefore, we must search for a smaller k in the lower half: [low, mid - 1].
            high = mid - 1 

    # After the binary search loop terminates, 'ans' holds the maximum value of k for which `is_subsequence(k)` returned True.
    # The overall time complexity is dominated by the binary search calls to `is_subsequence`.
    # Total time complexity: O(log m * (n + k)) = O(log m * (n + m)). Since m <= n, this is O(n log m) or O(n log n).
    # Space complexity: O(m) in the worst case for storing the `removed_indices` set inside `is_subsequence`.
    return ans