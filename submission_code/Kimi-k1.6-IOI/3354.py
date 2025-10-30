class Solution:
    def minimizeStringValue(self, s: str) -> str:
        initial_counts = [0] * 26
        k = 0
        for c in s:
            if c == '?':
                k += 1
            else:
                initial_counts[ord(c) - ord('a')] += 1
        
        added_counts = [0] * 26
        for _ in range(k):
            best_char = 0
            min_val = initial_counts[0] + added_counts[0]
            for i in range(1, 26):
                current_val = initial_counts[i] + added_counts[i]
                if current_val < min_val or (current_val == min_val and i < best_char):
                    min_val = current_val
                    best_char = i
            added_counts[best_char] += 1
        
        chars_to_insert = []
        for i in range(26):
            chars_to_insert.extend([chr(ord('a') + i)] * added_counts[i])
        chars_to_insert.sort()
        
        s_list = list(s)
        insert_ptr = 0
        for i in range(len(s_list)):
            if s_list[i] == '?':
                s_list[i] = chars_to_insert[insert_ptr]
                insert_ptr += 1
        
        return ''.join(s_list)