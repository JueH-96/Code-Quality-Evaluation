class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Count the frequency of each letter in the string excluding '?'
        letter_counts = [0] * 26
        question_marks = 0
        for char in s:
            if char == '?':
                question_marks += 1
            else:
                letter_counts[ord(char) - ord('a')] += 1
        
        # Sort the letters by their frequency (ascending order)
        sorted_letters = sorted([(count, letter) for letter, count in enumerate(letter_counts)], key=lambda x: x[0])
        
        # Replace '?' with the letters having the lowest frequency
        result = list(s)
        index = 0
        for i, char in enumerate(s):
            if char == '?':
                result[i] = chr(ord('a') + sorted_letters[index][1])
                sorted_letters[index] = (sorted_letters[index][0] + 1, sorted_letters[index][1])
                if sorted_letters[index][0] == question_marks:
                    index += 1
        
        return ''.join(result)