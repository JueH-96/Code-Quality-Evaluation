class Solution:
    def calculateScore(self, s: str) -> int:
        # Dictionary to store the mirror of each letter
        mirror = {chr(i): chr(25 - i + ord('a')) for i in range(26)}

        # Dictionary to store the first occurrence of each letter
        first_occurrence = {}

        # List to store the score for each index
        score = [0] * len(s)

        total_score = 0

        for i in range(len(s)):
            if s[i] in first_occurrence:
                j = first_occurrence[s[i]]
                score[i] = i - j
                total_score += score[i]
                del first_occurrence[s[i]]
            else:
                first_occurrence[mirror[s[i]]] = i

        return total_score