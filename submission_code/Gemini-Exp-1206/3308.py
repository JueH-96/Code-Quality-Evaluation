class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        last_occurrence = {}
        for i, char in enumerate(s):
            last_occurrence[char] = i
        
        max_last_occurrence = 0
        for char in last_occurrence:
            max_last_occurrence = max(max_last_occurrence, last_occurrence[char])
        
        result = ""
        for i, char in enumerate(s):
            if last_occurrence[char] == max_last_occurrence:
                result += char
        
        result_list = sorted(result, key=lambda char: last_occurrence[char])
        return "".join(result_list)