class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        banned_set = set(bannedWords)
        word_count = {}
        for word in message:
            if word in banned_set:
                word_count[word] = word_count.get(word, 0) + 1
                if word_count[word] >= 2:
                    return True
        return False