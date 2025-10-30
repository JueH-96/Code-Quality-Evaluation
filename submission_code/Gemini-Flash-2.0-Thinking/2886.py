class Solution:
    def finalString(self, s: str) -> str:
        current_string = ""
        for char in s:
            if char == 'i':
                current_string = current_string[::-1]
            else:
                current_string += char
        return current_string