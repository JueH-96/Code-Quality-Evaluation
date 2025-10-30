class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Count frequency of fixed characters
        freq = [0] * 26
        q_count = 0
        for c in s:
            if c == '?':
                q_count += 1
            else:
                freq[ord(c) - ord('a')] += 1
        
        # Find characters to replace ? with
        replacements = []
        for _ in range(q_count):
            min_freq = float('inf')
            min_char = 'a'
            for i in range(26):
                if freq[i] < min_freq:
                    min_freq = freq[i]
                    min_char = chr(i + ord('a'))
            replacements.append(min_char)
            freq[ord(min_char) - ord('a')] += 1
            
        # Sort replacements to ensure lexicographically smallest result
        replacements.sort()
        
        # Build result string
        result = []
        j = 0
        for c in s:
            if c == '?':
                result.append(replacements[j])
                j += 1
            else:
                result.append(c)
                
        return ''.join(result)