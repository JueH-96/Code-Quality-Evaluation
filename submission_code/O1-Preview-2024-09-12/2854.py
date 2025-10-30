class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        from collections import defaultdict

        dp = {}
        first_word = words[0]
        dp[(first_word[0], first_word[-1])] = len(first_word)

        for i in range(1, len(words)):
            word = words[i]
            dp_next = {}
            for (first_char, last_char), total_len in dp.items():
                # Option 1: join(str_i-1, word)
                new_first = first_char
                new_last = word[-1]
                if last_char == word[0]:
                    len_increment = len(word) - 1
                else:
                    len_increment = len(word)
                new_len = total_len + len_increment
                key = (new_first, new_last)
                if key not in dp_next or new_len < dp_next[key]:
                    dp_next[key] = new_len

                # Option 2: join(word, str_i-1)
                new_first = word[0]
                new_last = last_char
                if word[-1] == first_char:
                    len_increment = len(word) - 1
                else:
                    len_increment = len(word)
                new_len = total_len + len_increment
                key = (new_first, new_last)
                if key not in dp_next or new_len < dp_next[key]:
                    dp_next[key] = new_len
            dp = dp_next

        return min(dp.values())