class Solution:
    def minimumChairs(self, s: str) -> int:
        # Initialize the count of chairs and the count of people in the waiting room
        chairs = 0
        people = 0

        # Iterate over the string
        for i in range(len(s)):
            # If the person enters the waiting room
            if s[i] == 'E':
                # If there are no chairs available, increment chairs and people
                if chairs == 0:
                    chairs += 1
                    people += 1
                # If there are chairs available, increment people
                else:
                    people += 1
            # If the person leaves the waiting room
            else:
                # Decrement people
                people -= 1

        # Return the minimum number of chairs needed
        return max(0, chairs - people)