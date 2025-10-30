class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Alice has a winning move if and only if there is at least one vowel in s.
        # If the total number of vowels is odd, she can delete the entire string immediately.
        # If it is even but non‐zero, she can always delete one vowel at a time and
        # force a win by parity arguments—Bob cannot eliminate all vowels or avoid
        # eventually being passed a single‐vowel position where Alice wins.
        vowels = set("aeiou")
        for ch in s:
            if ch in vowels:
                return True
        return False