class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        for char in target:
            result.append(current)
            current += "a"
            result.append(current)
            
            for i in range(len(current)-1, -1, -1):
                if current[i] != "z":
                    current = current[:i] + chr(ord(current[i])+1) + current[i+1:]
                    result.append(current)
                    break
                else:
                    current = current[:i] + "a" + current[i+1:]
        
        result.append(current)
        return result