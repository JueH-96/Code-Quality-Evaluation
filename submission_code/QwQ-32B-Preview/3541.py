from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        banned_set = set(bannedWords)
        message_set = set(message)
        return len(message_set.intersection(banned_set)) >= 2