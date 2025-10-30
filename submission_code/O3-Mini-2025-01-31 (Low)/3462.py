class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Explanation:
        # In this game, on her turn Alice is required to remove a substring 
        # that has an odd number of vowels. Notice that if s contains at least one vowel,
        # then the substring consisting of a single vowel is a valid move for Alice.
        # Hence if s has at least one vowel, Alice can always make the first move.
        #
        # In fact, one can prove that if s contains no vowel then Alice can never move (since any substring
        # has 0 vowels which is even), and thus loses.
        # Otherwise (if there is a vowel in s), Alice can always choose a valid move and force a win.
        #
        # For example:
        #   s = "leetcoder" has vowels, so Alice starts by making a valid move.
        #   s = "bbcd" has no vowels, so Alice loses.
        #
        # Thus the answer is simply to check if there is any vowel in s.
        #
        # The set of vowels is:
        vowels = set("aeiou")
        for ch in s:
            if ch in vowels:
                return True
        return False