class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        
        # Special case: if numFriends is 1, return the entire word
        if numFriends == 1:
            return word
        
        # For each possible starting position, find the longest valid substring
        max_string = ""
        
        for start in range(n):
            # Calculate the maximum length of substring starting at 'start'
            # We need to leave at least (numFriends - 1) characters for other parts
            # Each of the other (numFriends - 1) parts needs at least 1 character
            remaining_chars_needed = numFriends - 1
            max_length = n - start - remaining_chars_needed + 1
            
            # The substring can be at most max_length characters long
            if max_length > 0:
                substring = word[start:start + max_length]
                if substring > max_string:
                    max_string = substring
        
        return max_string