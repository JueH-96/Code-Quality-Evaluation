class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        # Initialize the lexicographically largest string
        largest_string = ""
        
        # Iterate over all possible splits
        for i in range(len(word) - numFriends + 1):
            # Take the substring starting at i and of length numFriends
            current_split = word[i:i + numFriends]
            # Update the largest_string if the current_split is larger
            if current_split > largest_string:
                largest_string = current_split
        
        return largest_string