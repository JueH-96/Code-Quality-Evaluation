from typing import List

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        banned_words_set = set(bannedWords)
        count = 0
        for word in message:
            if word in banned_words_set:
                count += 1
        return count >= 2