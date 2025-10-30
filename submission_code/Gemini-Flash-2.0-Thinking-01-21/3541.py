class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        banned_words_set = set(bannedWords)
        banned_word_count = 0
        for word in message:
            if word in banned_words_set:
                banned_word_count += 1
        return banned_word_count >= 2