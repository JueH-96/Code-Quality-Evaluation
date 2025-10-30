class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        n = len(s)
        char_count = {}
        operation_number = [0] * n
        max_op = 0
        
        for i, char in enumerate(s):
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
            operation_number[i] = char_count[char]
            if operation_number[i] > max_op:
                max_op = operation_number[i]
        
        if max_op == 1:
            return s
        
        result = []
        for i in range(n):
            if operation_number[i] == max_op:
                result.append(s[i])
        
        return ''.join(result)