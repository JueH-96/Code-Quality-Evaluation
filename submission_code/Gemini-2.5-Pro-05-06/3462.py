class Solution:
  def doesAliceWin(self, s: str) -> bool:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    
    # Check if there is at least one vowel in the string s.
    # We don't need the total count, just the existence of at least one vowel.
    has_vowel = False
    for char_in_s in s:
      if char_in_s in vowels:
        has_vowel = True
        break
    
    # If there are no vowels in the string:
    # Alice needs to remove a non-empty substring that contains an odd number of vowels.
    # If `s` has no vowels, any non-empty substring of `s` will also have 0 vowels.
    # 0 is an even number. So, Alice cannot find a substring with an odd number of vowels.
    # Alice has no valid move and loses immediately.
    if not has_vowel:
      return False
    
    # If there is at least one vowel in the string (so `has_vowel` is true):
    # Alice can always make a move. For example, she can choose any single vowel
    # character `x` in `s`. The substring `x` has 1 vowel (which is odd).
    # Removing this substring `x` is a valid move for Alice.
    #
    # As shown in the reasoning above, Alice has a winning strategy if `has_vowel` is true.
    # The strategy relies on manipulating the parity of the vowel count of the string
    # presented to the opponent, ultimately leading to a win for Alice.
    # - If total vowels V(s) is odd, Alice removes s and wins.
    # - If total vowels V(s) is even and positive:
    #   - Alice removes 1 vowel, presents s' with V(s') = V(s)-1 (odd) to Bob.
    #   - If Bob faces s' (odd vowel count) and cannot move (e.g. s' is a single vowel), Bob loses.
    #   - If Bob moves, he removes sub' (even vowel count) from s'.
    #     The remaining s'' has V(s'') = V(s') - V(sub') = odd - even = odd vowels.
    #   - Alice faces s'' (odd vowel count), removes s'' and wins.
    #
    # Thus, if there's any vowel, Alice wins.
    return True