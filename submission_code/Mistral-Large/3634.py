class Solution:
    def calculateScore(self, s: str) -> int:
        # Dictionary to store the mirror of each letter
        mirror = {chr(i): chr(25 - i + 97) for i in range(97, 123)}

        # List to keep track of marked indices
        marked = [False] * len(s)

        # Total score
        score = 0

        # Iterate through the string
        for i in range(len(s)):
            if marked[i]:
                continue
            # Find the mirror character
            mirror_char = mirror[s[i]]
            # Find the closest unmarked index j such that j < i and s[j] is the mirror of s[i]
            for j in range(i-1, -1, -1):
                if not marked[j] and s[j] == mirror_char:
                    # Mark both indices i and j
                    marked[i] = marked[j] = True
                    # Add the value i - j to the total score
                    score += i - j
                    break

        return score