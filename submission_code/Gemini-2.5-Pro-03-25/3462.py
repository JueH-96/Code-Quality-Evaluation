import collections

class Solution:
  """
  Solves the Alice and Bob string game problem.

  Determines if Alice wins the game where players remove substrings
  based on the parity of vowels in the substring. Alice removes substrings
  with an odd number of vowels, Bob removes substrings with an even number
  of vowels. The first player unable to move loses. Alice starts first.
  Both players play optimally.

  The analysis of the game reveals a simple winning condition for Alice:
  Alice wins if and only if the initial string `s` contains at least one vowel.

  Detailed Analysis:

  1.  **If the string `s` contains no vowels:**
      - Alice's turn requires removing a non-empty substring with an odd number of vowels.
      - Since `s` has no vowels, any non-empty substring also has 0 vowels (an even number).
      - Alice cannot find a valid substring to remove.
      - Alice cannot make a move on her first turn and loses immediately.
      - In this case, the function should return `false`.

  2.  **If the string `s` contains at least one vowel:**
      - Alice needs to remove a substring with an odd number of vowels.
      - Alice can always make a valid move. For example, she can choose a single vowel character `v` from the string. The substring `v` has 1 vowel (odd). Removing this substring is a valid move.
      - Since Alice can always make the first move, we need to determine the winner assuming optimal play.
      - Consider the total number of vowels, `V`, in the string `s`.

      - **Case 2a: `V` is odd.**
          - Alice can choose to remove the entire string `s` as her first move.
          - The substring `s` has an odd number of vowels, making it a valid move for Alice.
          - After removing `s`, the string becomes empty.
          - It's Bob's turn. Bob faces an empty string. He cannot remove a non-empty substring.
          - Bob cannot make a move and loses. Alice wins.

      - **Case 2b: `V` is even and positive.**
          - Alice must make a move. She removes a substring `sub_A` which must contain an odd number of vowels (e.g., removing a single vowel).
          - The remaining string `s'` will have `V' = V - count_vowels(sub_A)` vowels. Since `V` is even and `count_vowels(sub_A)` is odd, `V'` must be odd.
          - It's Bob's turn. Bob faces the string `s'` which has an odd number of vowels (`V'` odd). Bob needs to remove a substring `sub_B` with an even number of vowels.
          - Bob cannot remove the entire string `s'` because `V'` is odd (Bob needs even).
          - Bob must make a move if possible. Bob can always find a substring with an even number of vowels (e.g., a single consonant has 0 vowels; two adjacent vowels have 2 vowels) unless `s'` consists of a single vowel. If `s'` was a single vowel, Bob would have lost immediately (Alice won on her first move). Assuming `s'` is not a single vowel, Bob removes `sub_B` (even vowel count `k >= 0`).
          - The remaining string `s''` will have `V'' = V' - k` vowels. Since `V'` is odd and `k` is even, `V''` must be odd.
          - It's Alice's turn. Alice faces the string `s''` which has an odd number of vowels (`V''` odd). Alice needs to remove a substring `sub_C` with an odd number of vowels.
          - Since `V''` is odd, Alice has a winning move available: she can choose to remove the entire string `s''`. This is a valid move because `s''` has an odd number of vowels.
          - After removing `s''`, the string becomes empty.
          - It's Bob's turn. Bob faces an empty string. He cannot make a move and loses. Alice wins.

  Conclusion: In all scenarios where the initial string `s` contains at least one vowel, Alice has a winning strategy. Therefore, Alice wins if and only if `s` contains at least one vowel.

  The implementation simply needs to check for the presence of any vowel in the string `s`.
  """
  def doesAliceWin(self, s: str) -> bool:
    """
    Checks if Alice wins the game based on the presence of vowels in the string.

    Args:
      s: The input string consisting of lowercase English letters.

    Returns:
      True if Alice wins (meaning the string contains at least one vowel),
      False otherwise (the string contains no vowels).
    """
    # Define the set of vowels for efficient checking.
    vowels = {'a', 'e', 'i', 'o', 'u'}
    
    # Iterate through each character in the string s.
    for char in s:
      # Check if the current character is a vowel.
      if char in vowels:
        # If a vowel is found, it means the total number of vowels is > 0.
        # Based on the game analysis, Alice has a winning strategy in this case.
        # We can stop searching and return True immediately.
        return True
        
    # If the loop completes without finding any vowels, it means the string s
    # consists only of consonants. The total vowel count is 0.
    # In this scenario, Alice cannot make her first move and loses.
    return False