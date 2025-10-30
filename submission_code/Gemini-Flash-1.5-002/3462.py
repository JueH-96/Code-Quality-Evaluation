class Solution:
    def doesAliceWin(self, s: str) -> bool:
        def count_vowels(substring):
            count = 0
            for char in substring:
                if char in "aeiou":
                    count += 1
            return count

        def can_win(current_s, turn):
            if not current_s:
                return False

            if turn == "Alice":
                for i in range(len(current_s)):
                    for j in range(i, len(current_s)):
                        substring = current_s[i:j+1]
                        if count_vowels(substring) % 2 != 0:
                            if not can_win(current_s[:i] + current_s[j+1:], "Bob"):
                                return True
                return False
            else:  # Bob's turn
                for i in range(len(current_s)):
                    for j in range(i, len(current_s)):
                        substring = current_s[i:j+1]
                        if count_vowels(substring) % 2 == 0:
                            if not can_win(current_s[:i] + current_s[j+1:], "Alice"):
                                return True
                return False

        return can_win(s, "Alice")