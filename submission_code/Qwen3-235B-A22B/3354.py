import heapq

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        cnt = [0] * 26
        k = 0
        for char in s:
            if char != '?':
                cnt[ord(char) - ord('a')] += 1
            else:
                k += 1
        
        heap = []
        for i in range(26):
            heapq.heappush(heap, (cnt[i], i, chr(ord('a') + i)))
        
        add = [0] * 26
        for _ in range(k):
            current_count, i, c = heapq.heappop(heap)
            add[i] += 1
            heapq.heappush(heap, (current_count + 1, i, c))
        
        added_chars = []
        for i in range(26):
            added_chars.extend([chr(ord('a') + i)] * add[i])
        
        res = []
        idx = 0
        for char in s:
            if char == '?':
                res.append(added_chars[idx])
                idx += 1
            else:
                res.append(char)
        
        return ''.join(res)