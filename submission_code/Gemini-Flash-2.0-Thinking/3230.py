class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n <= 1:
            return 0

        word_list = list(word)
        operations = 0

        for i in range(n - 1):
            char1 = word_list[i]
            char2 = word_list[i+1]

            if char1 == char2 or abs(ord(char1) - ord(char2)) == 1:
                operations += 1

                # Find a character to replace word_list[i+1]
                for j in range(ord('a'), ord('z') + 1):
                    new_char = chr(j)
                    if new_char != char1 and abs(ord(new_char) - ord(char1)) != 1:
                        word_list[i+1] = new_char
                        break

        return operations