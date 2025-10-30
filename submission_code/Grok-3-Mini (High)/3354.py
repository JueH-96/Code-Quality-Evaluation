import heapq

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        fixed_freq = [0] * 26
        question_indices = []
        for i in range(n):
            if s[i] == '?':
                question_indices.append(i)
            else:
                idx = ord(s[i]) - ord('a')
                fixed_freq[idx] += 1
        q = len(question_indices)
        if q == 0:
            return s
        
        freq = [f for f in fixed_freq]
        
        heap = []
        for i in range(26):
            heapq.heappush(heap, (freq[i], i))
        
        for _ in range(q):
            curr_freq, char_idx = heapq.heappop(heap)
            freq[char_idx] += 1
            heapq.heappush(heap, (freq[char_idx], char_idx))
        
        assign_count = [0] * 26
        for i in range(26):
            assign_count[i] = freq[i] - fixed_freq[i]
        
        assign_chars = []
        for i in range(26):
            char = chr(i + ord('a'))
            assign_chars.extend([char] * assign_count[i])
        assign_chars.sort()
        
        result = list(s)
        char_idx = 0
        for pos in question_indices:
            result[pos] = assign_chars[char_idx]
            char_idx += 1
        
        return ''.join(result)