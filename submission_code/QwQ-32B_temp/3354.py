import heapq

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        current_counts = [0] * 26
        for c in s:
            if c != '?':
                current_counts[ord(c) - ord('a')] += 1
        K = s.count('?')
        if K == 0:
            return s
        
        heap = []
        for i in range(26):
            cnt = current_counts[i]
            heapq.heappush(heap, (cnt, i))
        
        x = [0] * 26
        for _ in range(K):
            val, i = heapq.heappop(heap)
            x[i] += 1
            new_val = val + 1
            heapq.heappush(heap, (new_val, i))
        
        letters = []
        for i in range(26):
            letters += [chr(ord('a') + i)] * x[i]
        
        ptr = 0
        result = []
        for c in s:
            if c == '?':
                result.append(letters[ptr])
                ptr += 1
            else:
                result.append(c)
        return ''.join(result)