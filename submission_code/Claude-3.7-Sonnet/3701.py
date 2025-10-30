class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        if len(caption) < 3:
            return ""
        
        # Check if the original string is already a good caption
        good_chars = set()
        i = 0
        while i < len(caption):
            char = caption[i]
            count = 1
            j = i + 1
            while j < len(caption) and caption[j] == char:
                count += 1
                j += 1
            if count >= 3:
                good_chars.add(char)
            i = j
        
        if all(char in good_chars for char in set(caption)):
            return caption
        
        min_ops = float('inf')
        best_char = None
        
        for target_char in range(ord('a'), ord('z') + 1):
            target_char = chr(target_char)
            ops = sum(abs(ord(char) - ord(target_char)) for char in caption)
            
            if ops < min_ops:
                min_ops = ops
                best_char = target_char
                
        return best_char * len(caption) if best_char else ""