class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from collections import Counter
        
        # Count existing characters
        char_count = Counter(c for c in s if c != '?')
        
        # Count question marks
        q_count = s.count('?')
        
        # Determine characters to fill question marks
        to_fill = []
        for _ in range(q_count):
            next_char = min(range(ord('a'), ord('z') + 1), key=lambda x: (char_count[chr(x)], chr(x)))
            to_fill.append(chr(next_char))
            char_count[chr(next_char)] += 1
        
        # Sort the characters to fill
        to_fill.sort()
        
        # Construct the result string
        result = []
        fill_index = 0
        for c in s:
            if c == '?':
                result.append(to_fill[fill_index])
                fill_index += 1
            else:
                result.append(c)
        
        return ''.join(result)