class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        # Convert bannedWords to a set for O(1) lookup
        banned_set = set(bannedWords)
        
        # Count how many words in message are banned
        count = 0
        for word in message:
            if word in banned_set:
                count += 1
                # Early return if we found 2 banned words
                if count >= 2:
                    return True
        
        return False