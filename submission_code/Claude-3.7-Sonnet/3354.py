class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from collections import Counter
        
        # Count existing characters (excluding '?')
        letter_counts = Counter(s.replace('?', ''))
        
        # Determine the optimal replacements for each '?'
        replacements = []
        question_mark_count = s.count('?')
        
        for _ in range(question_mark_count):
            # Choose letter with lowest frequency so far (lexicographically smallest in case of ties)
            min_letter = min('abcdefghijklmnopqrstuvwxyz', key=lambda l: (letter_counts[l], l))
            replacements.append(min_letter)
            letter_counts[min_letter] += 1
        
        # Sort replacements to ensure lexicographically smallest result
        replacements.sort()
        
        # Replace each '?' with the appropriate character
        result = []
        replacement_index = 0
        
        for char in s:
            if char == '?':
                result.append(replacements[replacement_index])
                replacement_index += 1
            else:
                result.append(char)
        
        return ''.join(result)