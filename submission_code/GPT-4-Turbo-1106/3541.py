class Solution:
    def reportSpam(self, message: list[str], bannedWords: list[str]) -> bool:
        banned_set = set(bannedWords)
        spam_count = sum(word in banned_set for word in message)
        return spam_count >= 2