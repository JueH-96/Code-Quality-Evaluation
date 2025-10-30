class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        vowel_count = 0
        
        # Count the total number of vowels in the string s.
        for char in s:
            if char in vowels:
                vowel_count += 1
        
        # If the string contains no vowels (vowel_count is 0):
        # Alice needs to remove a non-empty substring with an ODD number of vowels.
        # Since there are no vowels in the string, any non-empty substring will have 0 vowels,
        # which is an EVEN number.
        # Therefore, Alice cannot make a valid move. In this scenario, Alice loses.
        if vowel_count == 0:
            return False
        
        # If the string contains at least one vowel (vowel_count > 0):
        # Alice can always win. Here's why:
        #
        # 1. Alice can always make a move:
        #    If vowel_count > 0, there is at least one vowel in 's'. Let 's[k]' be the first vowel.
        #    The prefix of 's' from index 0 up to 'k' (inclusive), i.e., s[0...k], will contain
        #    exactly one vowel (s[k]) and all preceding characters will be consonants.
        #    So, count_vowels(s[0...k]) will be 1, which is an ODD number.
        #    Alice can choose to remove this prefix s[0...k]. This is a valid move for her.
        #    Thus, Alice is never in a state where she cannot make a move (unless vowel_count was 0 initially).
        #
        # 2. Parity strategy:
        #    - If Alice's current string 's' has an ODD number of vowels (N_v_odd):
        #      Alice can simply choose to remove the entire string 's'. Since count_vowels(s) = N_v_odd,
        #      this is a valid move for Alice. The string becomes empty ("").
        #      Bob's turn on an empty string: Bob cannot make a non-empty move. Bob loses. Alice wins.
        #
        #    - If Alice's current string 's' has an EVEN number of vowels (N_v_even > 0):
        #      Alice must remove a substring with an ODD number of vowels. As explained in point 1,
        #      she can always find such a substring (e.g., the prefix ending at the first vowel, which has 1 vowel).
        #      When Alice removes a substring 'sub' with an ODD number of vowels, the new string s' will have
        #      (N_v_even - ODD_vowels) vowels. This results in an ODD number of vowels for s'.
        #      So, Alice always passes a string with an ODD number of vowels to Bob.
        #
        #    - Now it's Bob's turn, facing a string s' with an ODD number of vowels (N_v_odd_for_Bob).
        #      Bob needs to remove a substring 'sub_b' with an EVEN number of vowels.
        #      If Bob successfully makes a move, the new string s'' will have
        #      (N_v_odd_for_Bob - EVEN_vowels) vowels. This results in an ODD number of vowels for s''.
        #      So, Bob always passes a string with an ODD number of vowels back to Alice.
        #
        # This means that if Alice starts with an even number of vowels, she will make it odd for Bob,
        # and Bob will always return an odd number of vowels to Alice. Eventually, Alice will receive a string
        # with an odd number of vowels on her turn, and per the strategy above, she will remove the entire string,
        # leaving Bob with an empty string, causing Bob to lose.
        #
        # Therefore, if vowel_count is greater than 0, Alice always wins.
        return True