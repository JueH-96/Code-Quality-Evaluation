class Solution:
    def minimizeStringValue(self, s: str) -> str:
        original_counts = [0] * 26
        for c in s:
            if c != '?':
                original_counts[ord(c) - ord('a')] += 1
        
        num_q = s.count('?')
        add_counts = [0] * 26
        
        for _ in range(num_q):
            best_value = float('inf')
            best_char = None
            for i in range(26):
                current_value = original_counts[i] + add_counts[i]
                c = chr(ord('a') + i)
                if current_value < best_value:
                    best_value = current_value
                    best_char = c
                elif current_value == best_value:
                    if c < best_char:
                        best_char = c
            best_index = ord(best_char) - ord('a')
            add_counts[best_index] += 1
        
        # Generate added letters
        added_letters = []
        for i in range(26):
            added_letters.extend([chr(ord('a') + i)] * add_counts[i])
        added_letters.sort()
        
        # Replace '?' in the original string
        res = list(s)
        ptr = 0
        for i in range(len(res)):
            if res[i] == '?':
                res[i] = added_letters[ptr]
                ptr += 1
        
        return ''.join(res)