import heapq

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        freq = [0] * 26
        q_count = 0
        for ch in s:
            if ch != '?':
                freq[ord(ch) - ord('a')] += 1
            else:
                q_count += 1
        
        x = [0] * 26
        heap = []
        for idx in range(26):
            heapq.heappush(heap, (freq[idx], idx))
        
        for _ in range(q_count):
            cost, idx = heapq.heappop(heap)
            x[idx] += 1
            heapq.heappush(heap, (cost + 1, idx))
        
        replacements = []
        for idx in range(26):
            if x[idx] > 0:
                replacements.extend([chr(ord('a') + idx)] * x[idx])
        
        res = []
        rep_iter = iter(replacements)
        for ch in s:
            if ch == '?':
                res.append(next(rep_iter))
            else:
                res.append(ch)
        
        return "".join(res)