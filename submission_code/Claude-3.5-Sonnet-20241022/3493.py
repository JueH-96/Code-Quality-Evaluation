class Solution:
    def maxOperations(self, s: str) -> int:
        s = list(s)
        operations = 0
        
        while True:
            found = False
            for i in range(len(s)-1):
                if s[i] == '1' and s[i+1] == '0':
                    # Find position to move the 1
                    j = i + 1
                    while j < len(s) and s[j] == '0':
                        j += 1
                    
                    # Move the 1 to position j-1
                    s[i] = '0'
                    s[j-1] = '1'
                    operations += 1
                    found = True
                    break
                    
            if not found:
                break
                
        return operations