class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        banned_set = set(bannedWords)
        banned_count = 0
        for word in message:
            if word in banned_set:
                banned_count += 1
                if banned_count >= 2:
                    return True
        return False