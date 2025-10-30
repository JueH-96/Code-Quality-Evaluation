class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        for char in target:
            while len(current) < len(target) and (len(current) == 0 or current[-1] != char):
                if len(current) < len(target):
                    current += 'a'
                    result.append(current)
            
            while current[-1] != char:
                current = current[:-1] + chr(ord(current[-1]) + 1 if current[-1] != 'z' else 'a')
                result.append(current)
        
        return result