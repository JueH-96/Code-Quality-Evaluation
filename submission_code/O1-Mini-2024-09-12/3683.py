class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        
        for length in range(n, 0, -1):
            max_sub = ''
            for i in range(n - length + 1):
                j = i + length - 1
                assignable = False
                
                # Check the conditions for assignable split
                if i > 0 and j < n - 1 and numFriends >= 3:
                    assignable = True
                elif (i > 0 or j < n - 1) and numFriends >= 2:
                    assignable = True
                elif i == 0 and j == n - 1 and numFriends == 1:
                    assignable = True
                
                # Ensure the remaining characters can be split into the required number of friends
                if assignable and (n - length) >= (numFriends - 1):
                    current_sub = word[i:j+1]
                    if current_sub > max_sub:
                        max_sub = current_sub
                        
            if max_sub:
                return max_sub
        
        return ""