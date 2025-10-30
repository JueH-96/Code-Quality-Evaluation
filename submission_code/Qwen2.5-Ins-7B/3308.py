class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        last_occurrence = {char: -1 for char in 'abcdefghijklmnopqrstuvwxyz'}
        frequency = [0] * 26
        
        for i, char in enumerate(s):
            index = ord(char) - ord('a')
            frequency[index] += 1
            last_occurrence[char] = i
        
        max_freq = max(frequency)
        result = []
        
        for i, char in enumerate(s):
            if frequency[ord(char) - ord('a')] == max_freq:
                result.append(char)
        
        return ''.join(result)