class Solution:
    def minimizeStringValue(self, s: str) -> str:
        freq = [0] * 26
        k = 0
        for c in s:
            if c == '?':
                k += 1
            else:
                freq[ord(c) - ord('a')] += 1
        
        import heapq
        heap = [(count, i) for i, count in enumerate(freq)]
        heapq.heapify(heap)
        
        chosen = []
        for _ in range(k):
            count, c = heapq.heappop(heap)
            chosen.append(c)
            heapq.heappush(heap, (count + 1, c))
        
        chosen.sort()
        chars = [chr(c + ord('a')) for c in chosen]
        
        s_list = list(s)
        j = 0
        for i in range(len(s_list)):
            if s_list[i] == '?':
                s_list[i] = chars[j]
                j += 1
        
        return ''.join(s_list)