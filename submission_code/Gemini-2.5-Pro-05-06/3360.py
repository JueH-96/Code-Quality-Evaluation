import collections

class Solution:
  def minimumDeletions(self, word: str, k: int) -> int:
    # According to constraints: 1 <= word.length <= 10^5.
    # This means `word` is never empty.
    
    freq_map = collections.Counter(word)
    # `freqs` will store the frequencies of characters that are present in the original `word`.
    # Example: if word = "aabcaba", freq_map = {'a': 4, 'b': 2, 'c': 1}.
    # Then freqs = [4, 2, 1] (the order might vary but is not important).
    freqs = list(freq_map.values()) # freqs is guaranteed non-empty as word.length >= 1.
    
    # Optimization: If the original word is already k-special, no deletions are needed.
    # A word is k-special if for any two character types present in it,
    # their frequency difference is at most k. This is equivalent to:
    # (max_frequency - min_frequency) <= k, considering only characters present.
    if max(freqs) - min(freqs) <= k: # This check is safe as freqs is non-empty.
        return 0
        
    # Initialize min_total_deletions with the cost of deleting all characters.
    # This serves as an initial upper bound. An empty string is k-special.
    min_total_deletions = len(word) 

    # Determine the maximum frequency observed in the input string.
    # This helps limit the range of `x_target_min_freq` (our X) we need to check.
    max_f_val = max(freqs) # Safe as freqs is non-empty.

    # Iterate through all possible values for the lower bound `X` (here, x_target_min_freq)
    # of the target frequency window `[X, X+k]`.
    # `X` can range from 0 up to `max_f_val` (inclusive).
    # If `X` is chosen to be greater than `max_f_val`,
    # then for every character frequency `f_char` in `freqs`, `f_char < X`.
    # This implies all characters would be deleted. This scenario's cost (len(word))
    # is already covered by the initialization of `min_total_deletions`.
    for x_target_min_freq in range(max_f_val + 1): 
        current_deletions = 0
        # For each character type and its original frequency `f_char`:
        for f_char in freqs:
            if f_char < x_target_min_freq:
                # If `f_char` is less than the chosen minimum `X`,
                # all occurrences of this character must be deleted.
                # Characters kept must have frequency at least `x_target_min_freq`.
                current_deletions += f_char
            elif f_char > x_target_min_freq + k:
                # If `f_char` is greater than the chosen maximum `X+k`,
                # occurrences must be deleted to bring its frequency down to `X+k`.
                # Characters kept must have frequency at most `x_target_min_freq + k`.
                current_deletions += f_char - (x_target_min_freq + k)
            # else (i.e., x_target_min_freq <= f_char <= x_target_min_freq + k):
                # `f_char` is within the target window [X, X+k].
                # We can keep all occurrences of this character. No deletions needed for it.
        
        # Update the overall minimum deletions found so far.
        min_total_deletions = min(min_total_deletions, current_deletions)
    
    return min_total_deletions