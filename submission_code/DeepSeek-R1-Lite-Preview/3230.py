class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n == 0:
            return 0
        base = ord('a')
        # Precompute allowed transitions
        allowed = [[] for _ in range(26)]
        for c in range(26):
            for p in range(26):
                if abs(c - p) > 1:
                    allowed[c].append(p)
        # Initialize prev
        c0 = ord(word[0]) - base
        prev = [0 if c == c0 else 1 for c in range(26)]
        # Iterate through the string
        for i in range(1, n):
            curr = []
            for c in range(26):
                cost = 0 if word[i] == chr(c + base) else 1
                min_val = float('inf')
                for p in allowed[c]:
                    if prev[p] < min_val:
                        min_val = prev[p]
                curr.append(min_val + cost)
            prev = curr
        return min(prev)