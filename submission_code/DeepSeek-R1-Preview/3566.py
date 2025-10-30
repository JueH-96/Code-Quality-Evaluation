class Solution:
    def stringSequence(self, target: str) -> List[str]:
        res = []
        if not target:
            return res
        
        current = ""
        # Process the first character
        current += 'a'
        res.append(current)
        c0 = (ord(target[0]) - ord('a')) % 26
        for _ in range(c0):
            last = current[-1]
            new_char = chr((ord(last) - ord('a') + 1) % 26 + ord('a'))
            current = current[:-1] + new_char
            res.append(current)
        
        # Process the remaining characters
        for i in range(1, len(target)):
            current += 'a'
            res.append(current)
            ci = (ord(target[i]) - ord('a')) % 26
            for _ in range(ci):
                last = current[-1]
                new_char = chr((ord(last) - ord('a') + 1) % 26 + ord('a'))
                current = current[:-1] + new_char
                res.append(current)
        
        return res