class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        valid_strings = []
        
        def generate(current):
            if len(current) == n:
                valid_strings.append(current)
                return
            
            if current[-1] == '0':
                generate(current + '1')
            else:
                generate(current + '0')
                generate(current + '1')
        
        generate('0')
        generate('1')
        
        return valid_strings