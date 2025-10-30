class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        for char in target:
            while current[-1:] != char:
                if not current or current[-1] == 'z':
                    current += 'a'
                    result.append(current)
                else:
                    current = current[:-1] + chr(ord(current[-1]) + 1)
                    result.append(current)
            
            if len(current) < len(target):
                current += 'a'
                result.append(current)
        
        return result