from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        """
        Build the list of all intermediate strings that appear on the screen
        while typing `target` with the minimum possible key presses.

        Key-1  : append 'a'
        Key-2  : advance the last character one step in the alphabet (z→a)

        Strategy (provably optimal):
        • For every character in `target` we must press Key-1 once to add a new
          position (the only way to grow the string).  The freshly added
          character is always 'a'.
        • Afterwards we press Key-2 just enough times to turn that trailing
          'a' into the required target character.  Because the only allowed
          direction is forward through the alphabet, the minimum number of
          Key-2 presses is  (ord(target_char) - ord('a')).
        • We record the screen content after every key press.

        Complexity:
        Let n = len(target).  For each position we do at most 26 operations,
        so the total work is O(26·n) = O(n) and the list length never exceeds
        26·n ≤ 10400 (n ≤ 400), well within limits.
        """
        result = []
        screen = []                       # work with a mutable list of chars

        for ch in target:
            # Key-1: append 'a'
            screen.append('a')
            result.append(''.join(screen))

            # Key-2: increment to reach the target character
            steps = ord(ch) - ord('a')    # 0 … 25
            for _ in range(steps):
                last = screen[-1]
                # advance with wrap-around
                screen[-1] = 'a' if last == 'z' else chr(ord(last) + 1)
                result.append(''.join(screen))

        return result