import collections
import math # Import math, although not used in the final version, it's good practice if numerical operations are involved.

class Solution:
  """
  Solves the problem of finding the minimum deletions to make a word k-special.
  """
  def minimumDeletions(self, word: str, k: int) -> int:
    """
    Calculates the minimum number of characters to delete from a word 
    to make it k-special.

    A word is k-special if the absolute difference between the frequencies 
    of any two characters present in the word is less than or equal to k.
    That is, for any two characters `x` and `y` with `freq(x) > 0` and `freq(y) > 0` 
    in the resulting string, `|freq(x) - freq(y)| <= k`.

    Args:
        word: The input string. Consists only of lowercase English letters.
              Constraints: 1 <= word.length <= 10^5
        k: The maximum allowed frequency difference. 
           Constraints: 0 <= k <= 10^5

    Returns:
        The minimum number of deletions required to make the word k-special.
    """
    
    # 1. Count character frequencies in the input word.
    # This takes O(N) time, where N is the length of the word.
    freq_map = collections.Counter(word)
    
    # 2. Extract the frequency values (counts) of the characters that appear in the word.
    # We only need the counts, not the characters themselves.
    # This takes O(M) time, where M is the number of unique characters (M <= 26).
    counts = list(freq_map.values())
    
    # 3. Handle the base case: If there are 0 or 1 distinct character frequencies,
    # the word is already k-special (as the condition |freq(x) - freq(y)| <= k is trivial).
    # This check takes O(1) time.
    if len(counts) <= 1:
        return 0
        
    # 4. Get the unique non-zero frequencies present in the word.
    # These unique frequencies serve as candidates for the minimum frequency (`f_min`)
    # that could exist in an optimal k-special string derived from the original word.
    # Sorting is optional but can make debugging easier. It takes O(M log M) time, 
    # but since M <= 26, this is effectively O(1).
    unique_freqs = sorted(list(set(counts))) 
    
    # 5. Initialize `min_deletions` to the length of the word. 
    # This serves as an upper bound, representing the scenario where all characters are deleted.
    min_deletions = len(word) 
    
    # 6. Iterate through each unique frequency `f_min` found in the original word.
    # We test each `f_min` as a potential minimum frequency for the characters
    # we decide to keep in the final k-special string.
    # The outer loop runs U times, where U is the number of unique frequencies (U <= M <= 26).
    # This is O(1) iterations.
    for f_min in unique_freqs:
        current_deletions = 0
        # For a chosen `f_min`, the target frequency range for any kept character `c'` 
        # in the final string will be `f_min <= c' <= f_min + k`.
        
        # 7. Iterate through all the original character frequencies `c` from the `counts` list.
        # The inner loop runs M times (M <= 26). This is O(1) iterations.
        for c in counts:
            # Calculate the deletions needed for characters with original frequency `c`
            # to fit into the target range [f_min, f_min + k].
            
            # Case 1: The original frequency `c` is below the chosen minimum `f_min`.
            # To satisfy the condition `c' >= f_min`, all characters with this original
            # frequency must be deleted.
            if c < f_min:
                current_deletions += c
                
            # Case 2: The original frequency `c` is above the maximum allowed `f_min + k`.
            # To satisfy the condition `c' <= f_min + k`, we must delete characters
            # to reduce the frequency down to `f_min + k`. The number of deletions
            # needed for this frequency group is `c - (f_min + k)`.
            elif c > f_min + k:
                # Number of deletions = original count - target maximum count
                current_deletions += c - (f_min + k)
                
            # Case 3: `f_min <= c <= f_min + k`.
            # The original frequency `c` is already within the target range `[f_min, f_min + k]`.
            # We can potentially keep all `c` characters without violating the condition
            # for this choice of `f_min`. No deletions are forced by this range constraint.
            
        # 8. Update the overall minimum deletions found so far.
        # After calculating the total deletions needed for a specific `f_min`,
        # compare it with the minimum found previously and update if smaller.
        min_deletions = min(min_deletions, current_deletions)
            
    # 9. Return the overall minimum number of deletions required.
    return min_deletions

# Overall time complexity: O(N) due to the initial frequency counting. 
# The subsequent loops run in O(M*U) time, which is O(26*26) = O(1).
# Overall space complexity: O(M) for storing frequencies and unique frequencies. 
# Since M <= 26, this is O(1).