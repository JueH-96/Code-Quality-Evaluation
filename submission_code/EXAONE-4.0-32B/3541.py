class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        bannedSet = set(bannedWords)
        count = 0
        for word in message:
            if word in bannedSet:
                count += 1
                if count >= 2:
                    return True
        return False