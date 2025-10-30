class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        
        # Handle first character (can only use key 1 followed by key 2)
        current = 'a'
        result.append(current)
        
        # Get to first target character
        while current != target[0]:
            current = chr((ord(current) - ord('a') + 1) % 26 + ord('a'))
            result.append(current)
            
        # For remaining characters in target
        for i in range(1, len(target)):
            # First append 'a' using key 1
            current += 'a'
            result.append(current)
            
            # Use key 2 until we get desired character
            while current[-1] != target[i]:
                current = current[:-1] + chr((ord(current[-1]) - ord('a') + 1) % 26 + ord('a'))
                result.append(current)
                
        return result