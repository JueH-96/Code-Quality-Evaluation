class Solution:
    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        # Convert bannedWords to a set for O(1) average time complexity lookups
        banned_set = set(bannedWords)
        
        # Initialize a counter for banned words found in the message
        banned_count = 0
        
        # Iterate over each word in the message
        for word in message:
            # If the word is in the banned set, increment the counter
            if word in banned_set:
                banned_count += 1
                # If we have found at least two banned words, return True
                if banned_count >= 2:
                    return True
        
        # If less than two banned words are found, return False
        return False