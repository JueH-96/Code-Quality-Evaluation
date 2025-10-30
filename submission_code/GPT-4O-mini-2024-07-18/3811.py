class Solution:
    def reverseDegree(self, s: str) -> int:
        reverse_alphabet_value = {chr(i): 26 - (i - ord('a')) for i in range(ord('a'), ord('z') + 1)}
        total = 0
        
        for index, char in enumerate(s):
            position_in_string = index + 1  # 1-indexed
            position_in_reversed_alphabet = reverse_alphabet_value[char]
            total += position_in_string * position_in_reversed_alphabet
        
        return total