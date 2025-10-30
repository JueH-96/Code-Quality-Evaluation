class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        s = list(s)
        counts = {}
        for i in range(n):
            if s[i] == '?':
                min_char = 'a'
                min_cost = float('inf')
                for char_code in range(ord('a'), ord('z') + 1):
                    char = chr(char_code)
                    cost = counts.get(char, 0)
                    if cost < min_cost:
                        min_cost = cost
                        min_char = char
                    elif cost == min_cost and char < min_char:
                        min_char = char
                
                s[i] = min_char
                counts[min_char] = counts.get(min_char, 0) + 1
            else:
                counts[s[i]] = counts.get(s[i], 0) + 1
        return "".join(s)