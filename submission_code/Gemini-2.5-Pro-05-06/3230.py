import math

class Solution:
  def removeAlmostEqualCharacters(self, word: str) -> int:
    n = len(word)
    # If the word has 0 or 1 character, no adjacent characters exist, so 0 operations.
    if n <= 1:
      return 0

    operations = 0
    # i is the index of the current character word[i].
    # We check if word[i] and its predecessor word[i-1] are almost-equal.
    i = 1 
    while i < n:
      # Get integer values (0-25) for characters for easier comparison.
      char_i_val = ord(word[i]) - ord('a')
      char_prev_val = ord(word[i-1]) - ord('a')

      # Check if word[i] and word[i-1] are almost-equal.
      # Two characters are almost-equal if their values are the same or adjacent in the alphabet.
      # This means their ordinal values differ by at most 1.
      if abs(char_i_val - char_prev_val) <= 1:
        # An operation is required because word[i] and word[i-1] are almost-equal.
        operations += 1
        
        # We conceptually change word[i]. Let this new character be c'.
        # c' is chosen such that it's not almost-equal to word[i-1].
        # Importantly, c' can also be chosen so it's not almost-equal to the
        # original character word[i+1] (if i+1 < n). This is because there are
        # 26 possible lowercase English letters. word[i-1] makes at most 3 characters
        # forbidden for c'. word[i+1] (if it exists) also makes at most 3 characters
        # forbidden for c'. Even if these sets of forbidden characters are disjoint,
        # at most 6 characters are forbidden. Since 26 > 6, a valid c' always exists.
        #
        # Because this new c' (representing the modified word[i]) is not almost-equal
        # to word[i+1], the pair (modified word[i], word[i+1]) will satisfy the condition.
        # Therefore, we don't need to check word[i+1] against its predecessor (the modified word[i]).
        # The next pair that needs checking for a potential conflict is (word[i+1], word[i+2]).
        # Our loop variable 'i' points to the right character of the pair currently being checked.
        # If we are at 'i' (checking pair (word[i-1], word[i])), and we modify word[i],
        # the next check should effectively be for word[i+2] against word[i+1].
        # So, we advance 'i' by 2.
        i += 2
      else:
        # word[i] and word[i-1] are not almost-equal.
        # No operation is needed for word[i] based on its relation to word[i-1].
        # We move to check the next character, word[i+1], against word[i].
        i += 1
    
    return operations