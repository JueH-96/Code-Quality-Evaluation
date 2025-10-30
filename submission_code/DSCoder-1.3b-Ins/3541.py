class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        bannedSet = set(bannedWords)
        for word in message:
            if word in bannedSet:
                return True
        return False