from typing import List

class Solution:
  def maximumNumberOfStringPairs(self, words: List[str]) -> int:
    count = 0
    # This set will store strings that we have encountered
    # and are candidates for being the first element (words[i]) of a pair.
    seen_strings = set()

    # We iterate through the words. Each `current_word` can be thought of as `words[j]`.
    for current_word in words:
      # Calculate the reverse of the current word.
      # If `current_word` is `words[j]`, then `reversed_current_word` is `rev(words[j])`.
      # Since words[i].length == 2, current_word[::-1] is efficient.
      # e.g., if current_word is "ab", reversed_current_word is "ba".
      # if current_word is "aa" (a palindrome), reversed_current_word is "aa".
      reversed_current_word = current_word[::-1] 
      
      # The problem defines a pair (words[i], words[j]) if:
      # 1. words[i] == reversed(words[j])
      # 2. 0 <= i < j < words.length
      
      # In our loop, `current_word` corresponds to `words[j]`.
      # `reversed_current_word` corresponds to `reversed(words[j])`.
      # We need to check if an earlier string `words[i]` (where i < j)
      # exists such that `words[i] == reversed_current_word`.
      # The `seen_strings` set contains words `words[k]` (for k < j) that were
      # processed earlier and did not form a pair yet.
      if reversed_current_word in seen_strings:
        # A match is found.
        # `reversed_current_word` (which is `reversed(words[j])`) was an earlier string `words[i]`.
        # So, the condition `words[i] == reversed(words[j])` is satisfied.
        # The condition `i < j` is satisfied because `words[i]` (i.e. `reversed_current_word`)
        # was added to `seen_strings` during an earlier iteration.
        count += 1
        
        # `current_word` (i.e., `words[j]`) has now formed a pair. It should not be added to `seen_strings`,
        # effectively marking it as "used".
        # The `words[i]` (which is `reversed_current_word`) is also "used".
        # While it remains in `seen_strings`, it won't cause issues because all strings in `words` are distinct.
        # If another `future_word` later has its reverse equal to this `words[i]`,
        # then `future_word` would have to be identical to `current_word`, which is impossible.
      else:
        # `reversed_current_word` is not found in `seen_strings`.
        # This means no `words[i]` (with `i < j`) exists such that `words[i] == reversed(words[j])`.
        # So, the `current_word` does not complete a pair with an earlier string.
        # Add `current_word` to `seen_strings`. It might become `words[i]` for a future pair.
        seen_strings.add(current_word)
            
    return count