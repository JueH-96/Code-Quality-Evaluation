class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        max_string = ""
        
        # We need to split the word into numFriends parts
        # The maximum length of a single part can be n - (numFriends - 1)
        # because we need at least one character for each friend.
        
        # Iterate over all possible starting positions for the first part
        for i in range(n - numFriends + 1):
            # The first part can be from 1 to n - (numFriends - 1) characters long
            for j in range(1, n - i - (numFriends - 1) + 1):
                # Take the first part
                part = word[i:i+j]
                # Update the max_string if this part is lexicographically larger
                if part > max_string:
                    max_string = part
        
        return max_string