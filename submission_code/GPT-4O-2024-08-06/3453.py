class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        # Generate all binary strings of length n
        result = []
        
        def generate(current):
            if len(current) == n:
                result.append(current)
                return
            
            # If the last character is '0', we must add '1' next
            if len(current) == 0 or current[-1] == '1':
                generate(current + '0')
                generate(current + '1')
            else:
                generate(current + '1')
        
        generate("")
        return result