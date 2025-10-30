class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        for char in target:
            if not current:
                result.append("a")
                current = "a"
            else:
                while current[-1] != char:
                    current = current[:-1] + chr(((ord(current[-1]) - ord('a') + 1) % 26) + ord('a'))
                    result.append(current)
                
            if len(current) < len(target) and current[-1] == 'a' and len(current) < len(target):
              
              if len(current) < len(target) and target[len(current)] == 'a':
                current += 'a'
                result.append(current)
              elif len(current) < len(target) and target[len(current)] != 'a':
                pass
            elif len(current) < len(target) and current[-1] != 'a' and len(current) < len(target):
              if len(current) < len(target) and target[len(current)] == 'a':
                current += 'a'
                result.append(current)
              elif len(current) < len(target) and target[len(current)] != 'a':
                pass
        return result