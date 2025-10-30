class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        banned_count = 0
        banned_set = set(bannedWords)
        for word in message:
            if word in banned_set:
                banned_count += 1
        return banned_count >= 2