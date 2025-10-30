class Solution:
    def possibleStringCount(self, word: str) -> int:
        possible_strings = {word}
        n = len(word)

        for i in range(n - 1):
            if word[i] == word[i + 1]:
                char_to_remove = word[i]

                # Find the start and end of the consecutive sequence
                start = i
                while start > 0 and word[start - 1] == char_to_remove:
                    start -= 1

                end = i
                while end < n - 1 and word[end + 1] == char_to_remove:
                    end += 1

                # Consider removing one occurrence of char_to_remove from this sequence
                original_word = list(word)
                del original_word[i]
                possible_strings.add("".join(original_word))

        return len(possible_strings)