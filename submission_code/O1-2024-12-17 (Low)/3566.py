class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""
        
        for ch in target:
            # Step 1: Append 'a'
            current += 'a'
            result.append(current)
            
            # Step 2: Increment last character until it matches ch
            distance = ord(ch) - ord('a')
            for _ in range(distance):
                # Change the last character by 1 in the alphabet (wrapping z->a if needed)
                last_char = current[-1]
                next_char = chr((ord(last_char) - ord('a') + 1) % 26 + ord('a'))
                current = current[:-1] + next_char
                result.append(current)
        
        return result