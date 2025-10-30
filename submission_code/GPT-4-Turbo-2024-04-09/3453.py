class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        valid_strings = []
        
        def is_valid(s):
            for i in range(len(s) - 1):
                if s[i] == '0' and s[i + 1] == '0':
                    return False
            return True
        
        def generate(current):
            if len(current) == n:
                if is_valid(current):
                    valid_strings.append(current)
                return
            
            generate(current + '0')
            generate(current + '1')
        
        generate('')
        return valid_strings