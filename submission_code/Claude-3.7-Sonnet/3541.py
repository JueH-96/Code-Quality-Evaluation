class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        # Find the intersection of message and bannedWords
        common_words = set(message).intersection(set(bannedWords))
        
        # Return true if at least 2 words from message are in bannedWords
        return len(common_words) >= 2