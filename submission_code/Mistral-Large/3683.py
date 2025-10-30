class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        def generate_splits(word, numFriends):
            if numFriends == 1:
                yield [word]
                return
            for i in range(1, len(word)):
                for rest in generate_splits(word[i:], numFriends - 1):
                    yield [word[:i]] + rest

        splits = generate_splits(word, numFriends)
        lex_largest = ""

        for split in splits:
            for part in split:
                if part > lex_largest:
                    lex_largest = part

        return lex_largest