class Solution:
    def minimizeStringValue(self, s: str) -> str:
        used = set()
        for ch in s:
            if ch != '?':
                used.add(ch)
                
        available = sorted([chr(ord('a') + i) for i in range(26) if chr(ord('a') + i) not in used])
        result = []
        avail_idx = 0
        for ch in s:
            if ch != '?':
                result.append(ch)
            else:
                if avail_idx < len(available):
                    chosen = available[avail_idx]
                    avail_idx += 1
                else:
                    chosen = 'a'
                result.append(chosen)
        return ''.join(result)