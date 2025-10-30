from typing import List
from collections import Counter

class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        """
        Determines if the message contains at least two spam words from the bannedWords list.

        :param message: List of words in the message
        :param bannedWords: List of banned words to check against
        :return: True if message is considered spam, False otherwise
        """
        message_count = Counter(message)
        banned_count = Counter(bannedWords)
        spam_count = 0
        
        for word, count in message_count.items():
            if word in banned_count:
                spam_count += count
        
        return spam_count >= 2