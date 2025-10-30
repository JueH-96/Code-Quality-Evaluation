class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        banned_set = set(bannedWords)
        message_set = set(message)
        common_words = banned_set & message_set
        return len(common_words) >= 2