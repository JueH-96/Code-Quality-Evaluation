class Solution:
    def compressedString(self, word: str) -> str:
        if not word:
            return ""
        
        groups = []
        current_char = word[0]
        count = 1
        
        for c in word[1:]:
            if c == current_char:
                count += 1
            else:
                groups.append((current_char, count))
                current_char = c
                count = 1
        groups.append((current_char, count))  # Add the last group
        
        result = []
        for char, total in groups:
            remaining = total
            while remaining > 0:
                chunk_size = min(remaining, 9)
                result.append(f"{chunk_size}{char}")
                remaining -= chunk_size
        
        return ''.join(result)