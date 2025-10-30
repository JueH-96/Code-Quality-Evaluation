class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        best_sub = None
        n = len(s)
        for i in range(n):
            count = 0
            for j in range(i, n):
                if s[j] == '1':
                    count += 1
                if count == k:
                    current = s[i:j+1]
                    if best_sub is None:
                        best_sub = current
                    else:
                        if len(current) < len(best_sub):
                            best_sub = current
                        elif len(current) == len(best_sub):
                            if current < best_sub:
                                best_sub = current
                elif count > k:
                    break
        return best_sub if best_sub is not None else ""