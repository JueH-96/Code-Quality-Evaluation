class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        keyboard = set('qwertyuiopQWERTYUIOP')
        word_indices = []
        for i, word in enumerate(words):
            if any(char in keyboard for char in word):
                word_indices.append(i)
        return word_indices