class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        k = numFriends
        # Calculate the maximum length of any split part
        L = n - k + 1
        # Initialize the maximum substring
        max_sub = ""
        # Iterate through all possible starting positions
        for i in range(n):
            # Take the substring starting at i with length up to L
            candidate = word[i:i+L]
            # Update the maximum substring if the current candidate is larger
            if candidate > max_sub:
                max_sub = candidate
        return max_sub