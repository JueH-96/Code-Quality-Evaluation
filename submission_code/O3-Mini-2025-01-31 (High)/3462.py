class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Notice that a move is defined by removing a substring whose number of vowels (a,e,i,o,u)
        # has a certain parity: odd for Alice and even for Bob.
        #
        # Key idea:
        #   · If the string has no vowels, then any substring has 0 vowels (even).
        #     So on her first move (which requires odd vowels) Alice has no valid move and loses.
        #
        #   · If there is at least one vowel in s then:
        #       - If the total number of vowels in s is odd, then the entire string is a valid move for Alice.
        #         She can remove all letters and win immediately.
        #
        #       - If the total number of vowels is even then Alice cannot take the entire string
        #         (because that would have an even number of vowels) so she must remove a substring
        #         that has an odd number of vowels. One simple valid move is to remove just one vowel.
        #         This reduces the total vowel count by an odd number, leaving an odd number overall.
        #         After Bob’s move (which subtracts an even number, leaving the parity unchanged),
        #         the entire remaining string will again have an odd number of vowels. At that point
        #         Alice can remove the whole string (since an odd sum is allowed) and win.
        #
        # Thus, as long as s contains at least one vowel, Alice can force a win.
        
        vowels = set("aeiou")
        for ch in s:
            if ch in vowels:
                return True
        return False